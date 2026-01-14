#!/usr/bin/env python3
"""
Plot BFG order parameter scan results from compute_bfg_order_parameters --scan-dir

Usage:
    python plot_bfg_scan_results.py <scan_results.h5> [output_dir]
    python plot_bfg_scan_results.py --scan-dir <dir>  # auto-find scan_results.h5
    python plot_bfg_scan_results.py --scan-dir <dir> --plot-bonds  # include bond visualizations

Reads the scan_results.h5 file and produces plots of all order parameters vs Jpm:
  - Translation order (m_translation)
  - Nematic order (XY, S+S-, SzSz, Heisenberg)
  - VBS order (XY dimer, Heisenberg dimer)

With --plot-bonds, also reads per-Jpm HDF5 files to visualize:
  - Spatially resolved bond expectations (S+S-, SzSz, Heisenberg S·S)
  - Bond orientation visualization for nematic analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import os
from pathlib import Path
import glob
import re
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm


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


def load_bond_data(h5_file):
    """Load spatially resolved bond data from per-Jpm HDF5 file"""
    bond_data = {}
    with h5py.File(h5_file, 'r') as f:
        if 'bonds' not in f:
            return None
        
        bonds = f['bonds']
        bond_data['positions'] = bonds['positions'][:]
        bond_data['edges'] = bonds['edges'][:]
        
        if 'spsm' in bonds:
            spsm = bonds['spsm'][:]
            # Handle structured array (complex stored as compound type)
            if spsm.dtype.names and 'r' in spsm.dtype.names:
                bond_data['spsm'] = spsm['r'] + 1j * spsm['i']
            else:
                bond_data['spsm'] = spsm
        
        if 'szsz' in bonds:
            bond_data['szsz'] = bonds['szsz'][:]
        
        if 'heisenberg' in bonds:
            bond_data['heisenberg'] = bonds['heisenberg'][:]
        
        if 'orientation' in bonds:
            bond_data['orientation'] = bonds['orientation'][:]
        
        # Get Jpm from attribute if available
        if 'jpm' in f.attrs:
            bond_data['jpm'] = f.attrs['jpm']
    
    return bond_data


def find_per_jpm_files(scan_dir):
    """Find all order_params_Jpm=*.h5 files in order_parameter_results/"""
    result_dir = Path(scan_dir) / 'order_parameter_results'
    if not result_dir.exists():
        result_dir = Path(scan_dir)
    
    pattern = str(result_dir / 'order_params_Jpm=*.h5')
    files = glob.glob(pattern)
    
    # Extract Jpm values and sort
    jpm_files = []
    for f in files:
        match = re.search(r'Jpm=([+-]?[0-9]*\.?[0-9]+)', f)
        if match:
            jpm = float(match.group(1))
            jpm_files.append((jpm, f))
    
    jpm_files.sort(key=lambda x: x[0])
    return jpm_files


def plot_bond_lattice(positions, edges, bond_values, ax, title='', cmap='coolwarm', 
                      vmin=None, vmax=None, orientations=None, show_sites=True):
    """
    Plot bond values on the lattice as colored lines.
    
    Parameters:
    -----------
    positions : array (n_sites, 2)
        Site positions
    edges : array (n_bonds, 2)
        Edge list (pairs of site indices)
    bond_values : array (n_bonds,)
        Values to plot on each bond (real part if complex)
    ax : matplotlib axis
    title : str
    cmap : str or colormap
    vmin, vmax : float
        Color scale limits
    orientations : array (n_bonds,), optional
        Bond orientations (0, 1, 2) for marker style
    show_sites : bool
        Whether to show site positions
    """
    # Take real part if complex
    if np.iscomplexobj(bond_values):
        bond_values = np.real(bond_values)
    
    # Create line segments
    segments = []
    for e, (i, j) in enumerate(edges):
        p1 = positions[i]
        p2 = positions[j]
        segments.append([p1, p2])
    
    # Set color limits
    if vmin is None:
        vmin = np.min(bond_values)
    if vmax is None:
        vmax = np.max(bond_values)
    
    # Handle symmetric colormap
    if vmin < 0 and vmax > 0:
        vlim = max(abs(vmin), abs(vmax))
        vmin, vmax = -vlim, vlim
    
    # Create colored line collection
    norm = Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=3)
    lc.set_array(bond_values)
    ax.add_collection(lc)
    
    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, shrink=0.7)
    
    # Plot sites
    if show_sites:
        ax.scatter(positions[:, 0], positions[:, 1], c='black', s=30, zorder=5)
    
    ax.set_xlim(positions[:, 0].min() - 0.5, positions[:, 0].max() + 0.5)
    ax.set_ylim(positions[:, 1].min() - 0.5, positions[:, 1].max() + 0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    
    return lc, cbar


def plot_bonds_for_jpm(jpm, h5_file, output_dir, title_prefix=""):
    """Plot all spatially resolved bond expectations for a single Jpm value"""
    
    bond_data = load_bond_data(h5_file)
    if bond_data is None:
        print(f"  No bond data in {h5_file}")
        return
    
    positions = bond_data['positions']
    edges = bond_data['edges']
    
    # Create multi-panel figure
    n_panels = 0
    if 'spsm' in bond_data:
        n_panels += 2  # real and imag
    if 'szsz' in bond_data:
        n_panels += 1
    if 'heisenberg' in bond_data:
        n_panels += 1
    
    if n_panels == 0:
        return
    
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_panels > 1 else [axes]
    
    fig.suptitle(f'{title_prefix}Bond Expectations at $J_{{\\pm}}$ = {jpm:.4f}', fontsize=14)
    
    ax_idx = 0
    
    if 'spsm' in bond_data:
        spsm = bond_data['spsm']
        # Real part
        plot_bond_lattice(positions, edges, np.real(spsm), axes[ax_idx],
                          title=r'Re$\langle S^+_i S^-_j \rangle$', cmap='coolwarm')
        ax_idx += 1
        
        # Imaginary part (often zero or very small)
        if np.max(np.abs(np.imag(spsm))) > 1e-10:
            plot_bond_lattice(positions, edges, np.imag(spsm), axes[ax_idx],
                              title=r'Im$\langle S^+_i S^-_j \rangle$', cmap='coolwarm')
        else:
            axes[ax_idx].set_visible(False)
        ax_idx += 1
    
    if 'szsz' in bond_data:
        plot_bond_lattice(positions, edges, bond_data['szsz'], axes[ax_idx],
                          title=r'$\langle S^z_i S^z_j \rangle$', cmap='coolwarm')
        ax_idx += 1
    
    if 'heisenberg' in bond_data:
        plot_bond_lattice(positions, edges, bond_data['heisenberg'], axes[ax_idx],
                          title=r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$', cmap='coolwarm')
        ax_idx += 1
    
    # Hide unused axes
    for i in range(ax_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    jpm_str = f"{jpm:+.4f}".replace('+', 'p').replace('-', 'm').replace('.', 'p')
    fname = f'{output_dir}/bonds_Jpm_{jpm_str}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_bond_orientation_by_type(jpm, h5_file, output_dir, title_prefix=""):
    """
    Plot bonds colored by orientation (0, 1, 2) with mean value per orientation.
    This helps visualize nematic (C3-breaking) order.
    """
    bond_data = load_bond_data(h5_file)
    if bond_data is None or 'orientation' not in bond_data:
        return
    
    positions = bond_data['positions']
    edges = bond_data['edges']
    orientations = bond_data['orientation']
    
    # Colors for orientations
    orient_colors = ['#e41a1c', '#377eb8', '#4daf4a']  # red, blue, green
    orient_labels = ['Type 0', 'Type 1', 'Type 2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{title_prefix}Bond Expectations by Orientation at $J_{{\\pm}}$ = {jpm:.4f}', fontsize=14)
    
    for ax, (key, label) in zip(axes, [('spsm', r'$\langle S^+S^- \rangle$'),
                                        ('szsz', r'$\langle S^zS^z \rangle$'),
                                        ('heisenberg', r'$\langle \mathbf{S} \cdot \mathbf{S} \rangle$')]):
        if key not in bond_data:
            ax.set_visible(False)
            continue
        
        vals = bond_data[key]
        if np.iscomplexobj(vals):
            vals = np.real(vals)
        
        # Plot each orientation type separately
        for otype in [0, 1, 2]:
            mask = orientations == otype
            if not np.any(mask):
                continue
            
            # Create segments for this orientation
            segments = []
            values = []
            for e, (i, j) in enumerate(edges):
                if mask[e]:
                    segments.append([positions[i], positions[j]])
                    values.append(vals[e])
            
            if len(segments) > 0:
                mean_val = np.mean(values)
                lc = LineCollection(segments, colors=orient_colors[otype], linewidths=3,
                                    label=f'{orient_labels[otype]}: mean={mean_val:.4f}')
                ax.add_collection(lc)
        
        ax.scatter(positions[:, 0], positions[:, 1], c='black', s=30, zorder=5)
        ax.set_xlim(positions[:, 0].min() - 0.5, positions[:, 0].max() + 0.5)
        ax.set_ylim(positions[:, 1].min() - 0.5, positions[:, 1].max() + 0.5)
        ax.set_aspect('equal')
        ax.set_title(label)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    jpm_str = f"{jpm:+.4f}".replace('+', 'p').replace('-', 'm').replace('.', 'p')
    fname = f'{output_dir}/bond_orientations_Jpm_{jpm_str}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_bond_evolution(jpm_files, output_dir, title_prefix=""):
    """
    Plot evolution of mean bond values per orientation across Jpm values.
    Shows how nematic order develops.
    """
    jpm_vals = []
    means_by_type = {'spsm': [[], [], []], 'szsz': [[], [], []], 'heisenberg': [[], [], []]}
    
    for jpm, h5_file in jpm_files:
        bond_data = load_bond_data(h5_file)
        if bond_data is None or 'orientation' not in bond_data:
            continue
        
        jpm_vals.append(jpm)
        orientations = bond_data['orientation']
        
        for key in ['spsm', 'szsz', 'heisenberg']:
            if key not in bond_data:
                for otype in [0, 1, 2]:
                    means_by_type[key][otype].append(np.nan)
                continue
            
            vals = bond_data[key]
            if np.iscomplexobj(vals):
                vals = np.real(vals)
            
            for otype in [0, 1, 2]:
                mask = orientations == otype
                if np.any(mask):
                    means_by_type[key][otype].append(np.mean(vals[mask]))
                else:
                    means_by_type[key][otype].append(np.nan)
    
    if len(jpm_vals) == 0:
        return
    
    jpm_vals = np.array(jpm_vals)
    orient_colors = ['#e41a1c', '#377eb8', '#4daf4a']
    orient_labels = ['Type 0', 'Type 1', 'Type 2']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{title_prefix}Bond Expectation Evolution by Orientation', fontsize=14)
    
    for ax, (key, ylabel) in zip(axes, [('spsm', r'$\langle S^+S^- \rangle$'),
                                         ('szsz', r'$\langle S^zS^z \rangle$'),
                                         ('heisenberg', r'$\langle \mathbf{S} \cdot \mathbf{S} \rangle$')]):
        for otype in [0, 1, 2]:
            vals = np.array(means_by_type[key][otype])
            if np.all(np.isnan(vals)):
                continue
            ax.plot(jpm_vals, vals, 'o-', color=orient_colors[otype], 
                    label=orient_labels[otype], markersize=5)
        
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bond_evolution_by_orientation.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/bond_evolution_by_orientation.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/bond_evolution_by_orientation.png")
    plt.close()
    
    # Also plot the anisotropy (max - min across orientations)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{title_prefix}Bond Anisotropy (C3 Breaking Measure)', fontsize=14)
    
    for ax, (key, ylabel) in zip(axes, [('spsm', r'$\Delta \langle S^+S^- \rangle$'),
                                         ('szsz', r'$\Delta \langle S^zS^z \rangle$'),
                                         ('heisenberg', r'$\Delta \langle \mathbf{S} \cdot \mathbf{S} \rangle$')]):
        all_means = np.array([means_by_type[key][o] for o in [0, 1, 2]])  # (3, n_jpm)
        if np.all(np.isnan(all_means)):
            ax.set_visible(False)
            continue
        
        anisotropy = np.nanmax(all_means, axis=0) - np.nanmin(all_means, axis=0)
        ax.plot(jpm_vals, anisotropy, 'o-', color='purple', markersize=5)
        ax.fill_between(jpm_vals, 0, anisotropy, alpha=0.3, color='purple')
        
        ax.set_xlabel(r'$J_{\pm}$')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bond_anisotropy.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/bond_anisotropy.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir}/bond_anisotropy.png")
    plt.close()


def plot_all_bonds(scan_dir, output_dir, title_prefix="", select_jpm=None):
    """
    Plot spatially resolved bond data from per-Jpm HDF5 files.
    
    Parameters:
    -----------
    scan_dir : str
        Directory containing the scan
    output_dir : str  
        Output directory for plots
    title_prefix : str
        Title prefix for plots
    select_jpm : list of float, optional
        Only plot these specific Jpm values. If None, plot all.
    """
    jpm_files = find_per_jpm_files(scan_dir)
    
    if not jpm_files:
        print("No per-Jpm HDF5 files found with bond data")
        return
    
    print(f"Found {len(jpm_files)} per-Jpm files with potential bond data")
    
    bond_dir = Path(output_dir) / 'bonds'
    os.makedirs(bond_dir, exist_ok=True)
    
    # Filter to selected Jpm values if specified
    if select_jpm is not None:
        selected = []
        for target in select_jpm:
            # Find closest match
            closest = min(jpm_files, key=lambda x: abs(x[0] - target))
            if abs(closest[0] - target) < 0.01:
                selected.append(closest)
        jpm_files = selected
    
    # Plot individual Jpm files
    for jpm, h5_file in jpm_files:
        plot_bonds_for_jpm(jpm, h5_file, bond_dir, title_prefix)
        plot_bond_orientation_by_type(jpm, h5_file, bond_dir, title_prefix)
    
    # Plot evolution across Jpm
    all_jpm_files = find_per_jpm_files(scan_dir)  # Use all for evolution plot
    if len(all_jpm_files) > 1:
        plot_bond_evolution(all_jpm_files, output_dir, title_prefix)


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
    parser.add_argument('--plot-bonds', '-b', action='store_true',
                        help='Plot spatially resolved bond expectations from per-Jpm HDF5 files')
    parser.add_argument('--select-jpm', type=float, nargs='+', default=None,
                        help='Only plot bond data for these specific Jpm values')
    
    args = parser.parse_args()
    
    # Find input file
    scan_dir = None
    if args.scan_dir:
        scan_dir = args.scan_dir
        h5_file = Path(args.scan_dir) / 'order_parameter_results' / 'scan_results.h5'
        if not h5_file.exists():
            h5_file = Path(args.scan_dir) / 'scan_results.h5'
    elif args.input:
        h5_file = Path(args.input)
        scan_dir = h5_file.parent.parent if 'order_parameter_results' in str(h5_file) else h5_file.parent
    else:
        # Try current directory
        h5_file = Path('scan_results.h5')
        scan_dir = Path('.')
        if not h5_file.exists():
            h5_file = Path('order_parameter_results/scan_results.h5')
            scan_dir = Path('.')
    
    if not h5_file.exists():
        print(f"Error: Cannot find scan_results.h5")
        print(f"  Tried: {h5_file}")
        print("\nUsage:")
        print("  python plot_bfg_scan_results.py <scan_results.h5>")
        print("  python plot_bfg_scan_results.py --scan-dir <dir>")
        print("  python plot_bfg_scan_results.py --scan-dir <dir> --plot-bonds")
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
    
    # Plot spatially resolved bond data if requested
    if args.plot_bonds and scan_dir:
        print("\nPlotting spatially resolved bond expectations...")
        plot_all_bonds(scan_dir, output_dir, title_prefix, args.select_jpm)
    
    return 0


if __name__ == '__main__':
    exit(main())
