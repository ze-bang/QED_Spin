#!/usr/bin/env python3
"""
Plot detailed BFG order parameter results for a single Jpm value.

Usage:
    python plot_bfg_single_jpm.py <order_params_Jpm=X.XXXX.h5> [output_dir]

This reads the full HDF5 output from compute_bfg_order_parameters with --save-full
and produces detailed visualizations:
  - 2D S(q) and S_D(q) structure factor maps
  - Bond-resolved dimer correlation matrices
  - Bond expectations on the lattice
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import h5py
import argparse
import os
from pathlib import Path


def load_results(h5_file):
    """Load all data from order_params HDF5 file"""
    data = {}
    attrs = {}
    
    with h5py.File(h5_file, 'r') as f:
        # Load datasets
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                arr = f[key][:]
                # Handle complex numbers stored as compound type
                if arr.dtype.names and 'r' in arr.dtype.names:
                    data[key] = arr['r'] + 1j * arr['i']
                else:
                    data[key] = arr
            elif isinstance(f[key], h5py.Group):
                # Handle groups like /bonds
                group_data = {}
                for subkey in f[key].keys():
                    arr = f[key][subkey][:]
                    if arr.dtype.names and 'r' in arr.dtype.names:
                        group_data[subkey] = arr['r'] + 1j * arr['i']
                    else:
                        group_data[subkey] = arr
                data[key] = group_data
        
        # Load attributes
        for key in f.attrs.keys():
            attrs[key] = f.attrs[key]
    
    return data, attrs


def plot_structure_factor_2d(data, attrs, output_dir, prefix=""):
    """Plot 2D S(q) and S_D(q) maps"""
    
    q_grid = data.get('q_grid_vals', None)
    if q_grid is None:
        return
    
    n_q = len(q_grid)
    
    # Create meshgrid for plotting
    Q1, Q2 = np.meshgrid(q_grid, q_grid, indexing='ij')
    
    # =========================================================================
    # Spin structure factor S(q)
    # =========================================================================
    if 'S_q_2d' in data:
        s_q = data['S_q_2d']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Real part
        ax = axes[0]
        im = ax.pcolormesh(Q1, Q2, np.real(s_q), shading='auto', cmap='RdBu_r')
        ax.set_xlabel('$q_1$ (r.l.u.)')
        ax.set_ylabel('$q_2$ (r.l.u.)')
        ax.set_title('Re[S(q)]')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # Absolute value
        ax = axes[1]
        im = ax.pcolormesh(Q1, Q2, np.abs(s_q), shading='auto', cmap='hot')
        ax.set_xlabel('$q_1$ (r.l.u.)')
        ax.set_ylabel('$q_2$ (r.l.u.)')
        ax.set_title('|S(q)|')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{prefix}Spin Structure Factor', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/S_q_2d.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/S_q_2d.png")
        plt.close()
    
    # =========================================================================
    # VBS structure factor S_D(q) - XY
    # =========================================================================
    if 'S_D_q_xy_2d' in data:
        s_d = data['S_D_q_xy_2d']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax = axes[0]
        im = ax.pcolormesh(Q1, Q2, np.real(s_d), shading='auto', cmap='RdBu_r')
        ax.set_xlabel('$q_1$ (r.l.u.)')
        ax.set_ylabel('$q_2$ (r.l.u.)')
        ax.set_title('Re[$S_D$(q)] - XY Dimer')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1]
        im = ax.pcolormesh(Q1, Q2, np.abs(s_d), shading='auto', cmap='hot')
        ax.set_xlabel('$q_1$ (r.l.u.)')
        ax.set_ylabel('$q_2$ (r.l.u.)')
        ax.set_title('|$S_D$(q)| - XY Dimer')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{prefix}VBS Structure Factor (XY Dimer)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/S_D_q_xy_2d.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/S_D_q_xy_2d.png")
        plt.close()
    
    # =========================================================================
    # VBS structure factor S_D(q) - Heisenberg
    # =========================================================================
    if 'S_D_q_heis_2d' in data:
        s_d = data['S_D_q_heis_2d']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax = axes[0]
        vmax = np.max(np.abs(s_d))
        im = ax.pcolormesh(Q1, Q2, s_d, shading='auto', cmap='RdBu_r', 
                           vmin=-vmax, vmax=vmax)
        ax.set_xlabel('$q_1$ (r.l.u.)')
        ax.set_ylabel('$q_2$ (r.l.u.)')
        ax.set_title('$S_D$(q) - Heisenberg Dimer')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        ax = axes[1]
        im = ax.pcolormesh(Q1, Q2, np.abs(s_d), shading='auto', cmap='hot')
        ax.set_xlabel('$q_1$ (r.l.u.)')
        ax.set_ylabel('$q_2$ (r.l.u.)')
        ax.set_title('|$S_D$(q)| - Heisenberg Dimer')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{prefix}VBS Structure Factor (Heisenberg Dimer)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/S_D_q_heis_2d.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/S_D_q_heis_2d.png")
        plt.close()
    
    # =========================================================================
    # Comparison: S(q) vs S_D(q)
    # =========================================================================
    if 'S_q_2d' in data and ('S_D_q_xy_2d' in data or 'S_D_q_heis_2d' in data):
        n_plots = 1 + ('S_D_q_xy_2d' in data) + ('S_D_q_heis_2d' in data)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        idx = 0
        ax = axes[idx]
        im = ax.pcolormesh(Q1, Q2, np.abs(data['S_q_2d']), shading='auto', cmap='hot')
        ax.set_xlabel('$q_1$')
        ax.set_ylabel('$q_2$')
        ax.set_title('|S(q)| - Spin')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        idx += 1
        
        if 'S_D_q_xy_2d' in data:
            ax = axes[idx]
            im = ax.pcolormesh(Q1, Q2, np.abs(data['S_D_q_xy_2d']), shading='auto', cmap='hot')
            ax.set_xlabel('$q_1$')
            ax.set_ylabel('$q_2$')
            ax.set_title('|$S_D$(q)| - XY Dimer')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
            idx += 1
        
        if 'S_D_q_heis_2d' in data:
            ax = axes[idx]
            im = ax.pcolormesh(Q1, Q2, np.abs(data['S_D_q_heis_2d']), shading='auto', cmap='hot')
            ax.set_xlabel('$q_1$')
            ax.set_ylabel('$q_2$')
            ax.set_title('|$S_D$(q)| - Heis Dimer')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)
        
        plt.suptitle(f'{prefix}Structure Factor Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/structure_factor_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/structure_factor_comparison.png")
        plt.close()


def plot_dimer_correlation_matrices(data, attrs, output_dir, prefix=""):
    """Plot bond-resolved dimer-dimer correlation matrices"""
    
    # XY dimer-dimer correlation
    if 'dimer_corr_xy' in data:
        corr = data['dimer_corr_xy']
        n_bonds = corr.shape[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Raw correlation
        ax = axes[0]
        im = ax.imshow(np.real(corr), cmap='RdBu_r', aspect='equal')
        ax.set_xlabel('Bond $b\'$')
        ax.set_ylabel('Bond $b$')
        ax.set_title('$\\langle D_b D_{b\'} \\rangle$ (XY)')
        plt.colorbar(im, ax=ax)
        
        # Connected correlation
        if 'connected_corr_xy' in data:
            conn = data['connected_corr_xy']
            ax = axes[1]
            vmax = np.max(np.abs(np.real(conn)))
            im = ax.imshow(np.real(conn), cmap='RdBu_r', aspect='equal',
                           vmin=-vmax, vmax=vmax)
            ax.set_xlabel('Bond $b\'$')
            ax.set_ylabel('Bond $b$')
            ax.set_title('$\\langle \\delta D_b \\delta D_{b\'} \\rangle$ (XY)')
            plt.colorbar(im, ax=ax)
        
        # Diagonal vs off-diagonal
        ax = axes[2]
        diag = np.real(np.diag(corr))
        ax.bar(range(n_bonds), diag, color='steelblue', alpha=0.7)
        ax.set_xlabel('Bond index')
        ax.set_ylabel('$\\langle D_b D_b \\rangle$')
        ax.set_title('Diagonal Elements (Self-correlation)')
        ax.axhline(np.mean(diag), color='red', linestyle='--', label=f'Mean: {np.mean(diag):.4f}')
        ax.legend()
        
        plt.suptitle(f'{prefix}XY Dimer-Dimer Correlations ({n_bonds} bonds)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dimer_corr_xy.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/dimer_corr_xy.png")
        plt.close()
    
    # Heisenberg dimer-dimer correlation
    if 'dimer_corr_heis' in data:
        corr = data['dimer_corr_heis']
        n_bonds = corr.shape[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Raw correlation
        ax = axes[0]
        im = ax.imshow(corr, cmap='RdBu_r', aspect='equal')
        ax.set_xlabel('Bond $b\'$')
        ax.set_ylabel('Bond $b$')
        ax.set_title('$\\langle (S \\cdot S)_b (S \\cdot S)_{b\'} \\rangle$')
        plt.colorbar(im, ax=ax)
        
        # Connected correlation
        if 'connected_corr_heis' in data:
            conn = data['connected_corr_heis']
            ax = axes[1]
            vmax = np.max(np.abs(conn))
            im = ax.imshow(conn, cmap='RdBu_r', aspect='equal',
                           vmin=-vmax, vmax=vmax)
            ax.set_xlabel('Bond $b\'$')
            ax.set_ylabel('Bond $b$')
            ax.set_title('Connected: $\\langle \\delta(S \\cdot S)_b \\delta(S \\cdot S)_{b\'} \\rangle$')
            plt.colorbar(im, ax=ax)
        
        # Diagonal
        ax = axes[2]
        diag = np.diag(corr)
        ax.bar(range(n_bonds), diag, color='darkorange', alpha=0.7)
        ax.set_xlabel('Bond index')
        ax.set_ylabel('$\\langle (S \\cdot S)_b^2 \\rangle$')
        ax.set_title('Diagonal Elements')
        ax.axhline(np.mean(diag), color='red', linestyle='--', label=f'Mean: {np.mean(diag):.4f}')
        ax.legend()
        
        plt.suptitle(f'{prefix}Heisenberg Dimer-Dimer Correlations ({n_bonds} bonds)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dimer_corr_heis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/dimer_corr_heis.png")
        plt.close()


def plot_lattice_bonds(data, attrs, output_dir, prefix=""):
    """Plot bond expectations on the lattice"""
    
    if 'bonds' not in data:
        return
    
    bonds = data['bonds']
    if 'positions' not in bonds or 'edges' not in bonds:
        return
    
    pos = bonds['positions']
    edges = bonds['edges']
    n_bonds = len(edges)
    
    # Get orientation colors
    orientations = bonds.get('orientation', np.zeros(n_bonds, dtype=int))
    orient_colors = ['#e41a1c', '#377eb8', '#4daf4a']  # Red, blue, green for α=0,1,2
    
    def plot_bonds_colored(ax, values, title, cmap='RdBu_r', symmetric=True):
        """Helper to plot bonds colored by a scalar value"""
        ax.scatter(pos[:, 0], pos[:, 1], s=100, c='black', zorder=5)
        
        # Add site labels
        for i, (x, y) in enumerate(pos):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', 
                       color='white', fontweight='bold', zorder=6)
        
        if symmetric:
            vmax = np.max(np.abs(values))
            vmin = -vmax
        else:
            vmin, vmax = np.min(values), np.max(values)
        
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        segments = []
        colors = []
        for b, (i, j) in enumerate(edges):
            segments.append([pos[i], pos[j]])
            colors.append(values[b])
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=3, zorder=1)
        lc.set_array(np.array(colors))
        ax.add_collection(lc)
        
        ax.set_aspect('equal')
        ax.set_title(title)
        return lc
    
    # =========================================================================
    # Figure 1: All bond expectations
    # =========================================================================
    n_plots = sum([
        'spsm' in bonds,
        'szsz' in bonds,
        'heisenberg' in bonds,
    ])
    
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        idx = 0
        
        if 'spsm' in bonds:
            vals = np.real(bonds['spsm'])
            lc = plot_bonds_colored(axes[idx], vals, '$\\langle S^+_i S^-_j \\rangle$')
            plt.colorbar(lc, ax=axes[idx])
            idx += 1
        
        if 'szsz' in bonds:
            vals = bonds['szsz']
            lc = plot_bonds_colored(axes[idx], vals, '$\\langle S^z_i S^z_j \\rangle$')
            plt.colorbar(lc, ax=axes[idx])
            idx += 1
        
        if 'heisenberg' in bonds:
            vals = bonds['heisenberg']
            lc = plot_bonds_colored(axes[idx], vals, '$\\langle S_i \\cdot S_j \\rangle$')
            plt.colorbar(lc, ax=axes[idx])
            idx += 1
        
        plt.suptitle(f'{prefix}Bond Expectations', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bond_expectations.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/bond_expectations.png")
        plt.close()
    
    # =========================================================================
    # Figure 2: Bond orientations (for kagome)
    # =========================================================================
    if np.max(orientations) > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(pos[:, 0], pos[:, 1], s=100, c='black', zorder=5)
        for i, (x, y) in enumerate(pos):
            ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center', 
                       color='white', fontweight='bold', zorder=6)
        
        for b, (i, j) in enumerate(edges):
            color = orient_colors[orientations[b] % 3]
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                   color=color, linewidth=2, zorder=1)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=orient_colors[0], linewidth=2, label='α=0'),
            Line2D([0], [0], color=orient_colors[1], linewidth=2, label='α=1'),
            Line2D([0], [0], color=orient_colors[2], linewidth=2, label='α=2'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_aspect('equal')
        ax.set_title(f'{prefix}Bond Orientations')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bond_orientations.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_dir}/bond_orientations.png")
        plt.close()


def print_summary(attrs, prefix=""):
    """Print summary of order parameters"""
    print(f"\n{prefix}Order Parameter Summary:")
    print("=" * 50)
    
    if 'm_translation' in attrs:
        print(f"  Translation:  m = {attrs['m_translation']:.6f}")
    
    if 'm_nematic' in attrs:
        print(f"  Nematic (XY): m = {attrs['m_nematic']:.6f}")
    if 'm_nematic_heisenberg' in attrs:
        print(f"  Nematic (Heis): m = {attrs['m_nematic_heisenberg']:.6f}")
    
    if 'm_vbs_xy' in attrs:
        print(f"  VBS (XY):     m = {attrs['m_vbs_xy']:.6f}")
    if 'm_vbs_heis' in attrs:
        print(f"  VBS (Heis):   m = {attrs['m_vbs_heis']:.6f}")
    elif 'm_vbs' in attrs:
        print(f"  VBS:          m = {attrs['m_vbs']:.6f}")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Plot detailed BFG order parameter results for a single Jpm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', help='Path to order_params_Jpm=X.XXXX.h5')
    parser.add_argument('output_dir', nargs='?', default=None,
                        help='Output directory for plots (default: same as input)')
    parser.add_argument('--title', '-t', type=str, default='',
                        help='Title prefix for plots')
    
    args = parser.parse_args()
    
    h5_file = Path(args.input)
    if not h5_file.exists():
        print(f"Error: File not found: {h5_file}")
        return 1
    
    # Extract Jpm from filename if possible
    jpm_str = ""
    if 'Jpm=' in h5_file.stem:
        jpm_str = h5_file.stem.split('Jpm=')[1].split('.h5')[0]
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = h5_file.parent / f'plots_Jpm={jpm_str}' if jpm_str else h5_file.parent / 'plots'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading: {h5_file}")
    data, attrs = load_results(h5_file)
    
    prefix = args.title + ' ' if args.title else ''
    if jpm_str:
        prefix += f'$J_{{\\pm}}$={jpm_str} '
    
    print_summary(attrs, prefix.replace('$', '').replace('{', '').replace('}', '').replace('\\pm', '±'))
    
    plot_structure_factor_2d(data, attrs, output_dir, prefix)
    plot_dimer_correlation_matrices(data, attrs, output_dir, prefix)
    plot_lattice_bonds(data, attrs, output_dir, prefix)
    
    print(f"\nAll plots saved to: {output_dir}/")
    return 0


if __name__ == '__main__':
    exit(main())
