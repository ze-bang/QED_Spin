#!/usr/bin/env python3
"""
Visualize lattice with bonds colored by dimer strength.
Supports both <S·S> (Heisenberg) and <S+S-> bond operators.

Usage:
    python plot_lattice_bonds.py <order_parameters.h5> [output_prefix]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import h5py
import sys
import os

def load_bond_data(h5_path):
    """Load bond data from compute_bfg_order_parameters output."""
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        if 'bonds' not in f:
            raise ValueError("No bond data found in HDF5 file. Run compute_bfg_order_parameters first.")
        
        bonds = f['bonds']
        data['positions'] = bonds['positions'][:]
        data['edges'] = bonds['edges'][:]
        
        if 'spsm' in bonds:
            spsm = bonds['spsm'][:]
            # Complex type stored as structured array
            if spsm.dtype.names:
                data['spsm'] = spsm['r'] + 1j * spsm['i']
            else:
                data['spsm'] = spsm
        
        if 'szsz' in bonds:
            data['szsz'] = bonds['szsz'][:]
            
        if 'heisenberg' in bonds:
            data['heisenberg'] = bonds['heisenberg'][:]
        
        if 'orientation' in bonds:
            data['orientation'] = bonds['orientation'][:]
        
        # Load nematic order parameters
        data['nematic'] = {}
        root = f['/']
        for bond_type in ['', '_spsm', '_szsz', '_heisenberg']:
            m_key = f'm_nematic{bond_type}'
            a_key = f'nematic_anisotropy{bond_type}'
            if m_key in root.attrs:
                name = bond_type.lstrip('_') if bond_type else 'xy'
                data['nematic'][name] = {
                    'm_nem': root.attrs[m_key],
                    'anisotropy': root.attrs[a_key]
                }
    
    return data

def plot_lattice_bonds(data, bond_type='heisenberg', ax=None, cmap='RdBu_r', 
                       show_sites=True, site_size=200, linewidth=8,
                       title=None, vmin=None, vmax=None):
    """
    Plot the lattice with bonds colored by their expectation value.
    
    Parameters:
    -----------
    data : dict
        Bond data from load_bond_data
    bond_type : str
        'heisenberg' for <S·S>, 'spsm' for <S+S->, 'szsz' for <SzSz>
    ax : matplotlib axis
        If None, creates new figure
    cmap : str
        Colormap name
    show_sites : bool
        Whether to show lattice sites
    site_size : float
        Size of site markers
    linewidth : float
        Width of bond lines
    title : str
        Plot title
    vmin, vmax : float
        Color scale limits
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    positions = data['positions']
    edges = data['edges']
    
    # Get bond values
    if bond_type == 'heisenberg':
        bond_vals = data.get('heisenberg', None)
        if bond_vals is None:
            raise ValueError("No Heisenberg bond data found")
        label = r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$'
    elif bond_type == 'spsm':
        bond_vals = data.get('spsm', None)
        if bond_vals is None:
            raise ValueError("No S+S- bond data found")
        bond_vals = np.real(bond_vals)  # Take real part
        label = r'$\langle S_i^+ S_j^- \rangle$'
    elif bond_type == 'szsz':
        bond_vals = data.get('szsz', None)
        if bond_vals is None:
            raise ValueError("No SzSz bond data found")
        label = r'$\langle S_i^z S_j^z \rangle$'
    else:
        raise ValueError(f"Unknown bond_type: {bond_type}")
    
    # Create line segments for bonds
    segments = []
    for i, (s1, s2) in enumerate(edges):
        p1 = positions[s1]
        p2 = positions[s2]
        segments.append([p1, p2])
    
    # Set color scale
    if vmin is None:
        vmin = np.min(bond_vals)
    if vmax is None:
        vmax = np.max(bond_vals)
    
    # Make symmetric around 0 if data spans both signs
    if vmin < 0 and vmax > 0:
        vmax_abs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vmax_abs, vmax_abs
    
    # Create LineCollection
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth, 
                        capstyle='round', zorder=1)
    lc.set_array(bond_vals)
    ax.add_collection(lc)
    
    # Add colorbar
    cbar = fig.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(label, fontsize=14)
    
    # Draw sites
    if show_sites:
        ax.scatter(positions[:, 0], positions[:, 1], s=site_size, c='white', 
                   edgecolors='black', linewidths=1.5, zorder=2)
        
        # Add site labels
        for i, (x, y) in enumerate(positions):
            ax.annotate(str(i), (x, y), ha='center', va='center', fontsize=8, 
                       fontweight='bold', zorder=3)
    
    # Set axis properties
    ax.set_aspect('equal')
    margin = 0.5
    ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    
    # Add bond value statistics
    stats_text = f'min: {np.min(bond_vals):.4f}\nmax: {np.max(bond_vals):.4f}\navg: {np.mean(bond_vals):.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return fig, ax


def create_comparison_plot(data, output_prefix, show=True):
    """Create a comparison plot with all bond types."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Heisenberg S·S
    if 'heisenberg' in data:
        plot_lattice_bonds(data, 'heisenberg', ax=axes[0], 
                          title=r'Heisenberg: $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$')
    
    # S+S-
    if 'spsm' in data:
        plot_lattice_bonds(data, 'spsm', ax=axes[1],
                          title=r'XY: $\langle S_i^+ S_j^- \rangle$', cmap='viridis')
    
    # SzSz
    if 'szsz' in data:
        plot_lattice_bonds(data, 'szsz', ax=axes[2],
                          title=r'Ising: $\langle S_i^z S_j^z \rangle$')
    
    plt.tight_layout()
    
    output_file = output_prefix + '_bond_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot to: {output_file}")


def plot_nematic_by_orientation(data, bond_type='heisenberg', ax=None, title=None):
    """
    Plot bond values grouped by orientation (for nematic order visualization).
    Kagome has 3 bond orientations (0, 1, 2).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.get_figure()
    
    positions = data['positions']
    edges = data['edges']
    orientations = data.get('orientation', None)
    
    if orientations is None:
        print("Warning: No orientation data found. Using default colors.")
        return plot_lattice_bonds(data, bond_type, ax=ax, title=title)
    
    # Get bond values
    if bond_type == 'heisenberg':
        bond_vals = data['heisenberg']
        label = r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$'
    elif bond_type == 'spsm':
        bond_vals = np.real(data['spsm'])
        label = r'$\langle S_i^+ S_j^- \rangle$'
    elif bond_type == 'szsz':
        bond_vals = data['szsz']
        label = r'$\langle S_i^z S_j^z \rangle$'
    else:
        raise ValueError(f"Unknown bond_type: {bond_type}")
    
    # Colors for each orientation
    orient_colors = ['#e41a1c', '#377eb8', '#4daf4a']  # red, blue, green
    orient_labels = [r'$\alpha=0$', r'$\alpha=1$', r'$\alpha=2$']
    
    # Group bonds by orientation
    for orient in [0, 1, 2]:
        mask = orientations == orient
        orient_edges = edges[mask]
        orient_vals = bond_vals[mask]
        
        if len(orient_edges) == 0:
            continue
        
        # Create line segments
        segments = []
        for (s1, s2), val in zip(orient_edges, orient_vals):
            p1 = positions[s1]
            p2 = positions[s2]
            segments.append([p1, p2])
        
        # Calculate line widths based on bond strength (stronger = thicker)
        widths = 3 + 10 * (np.abs(orient_vals) - np.min(np.abs(bond_vals))) / (np.max(np.abs(bond_vals)) - np.min(np.abs(bond_vals)) + 1e-10)
        
        # Draw bonds with this orientation
        lc = LineCollection(segments, colors=[orient_colors[orient]] * len(segments),
                           linewidths=widths, capstyle='round', zorder=1,
                           label=f'{orient_labels[orient]}: avg={np.mean(orient_vals):.4f}')
        ax.add_collection(lc)
    
    # Draw sites
    ax.scatter(positions[:, 0], positions[:, 1], s=200, c='white', 
               edgecolors='black', linewidths=1.5, zorder=2)
    
    # Add site labels
    for i, (x, y) in enumerate(positions):
        ax.annotate(str(i), (x, y), ha='center', va='center', fontsize=8, 
                   fontweight='bold', zorder=3)
    
    # Set axis properties
    ax.set_aspect('equal')
    margin = 0.5
    ax.set_xlim(positions[:, 0].min() - margin, positions[:, 0].max() + margin)
    ax.set_ylim(positions[:, 1].min() - margin, positions[:, 1].max() + margin)
    ax.legend(loc='upper right', fontsize=10)
    
    # Add nematic order info
    if 'nematic' in data and bond_type.replace('spsm', 'spsm') in data['nematic']:
        nem_key = bond_type if bond_type != 'spsm' else 'spsm'
        if nem_key == 'heisenberg':
            nem_key = 'heisenberg'
        elif nem_key == 'spsm':
            nem_key = 'spsm'
        elif nem_key == 'szsz':
            nem_key = 'szsz'
        
        if nem_key in data['nematic']:
            nem = data['nematic'][nem_key]
            nem_text = f"$m_{{nem}}$ = {nem['m_nem']:.4f}\nAnisotropy = {nem['anisotropy']:.4f}"
            ax.text(0.02, 0.98, nem_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    if title:
        ax.set_title(title, fontsize=14)
    
    return fig, ax


def create_nematic_comparison_plot(data, output_prefix):
    """Create a plot comparing nematic order for different bond types."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    bond_types = [
        ('heisenberg', r'Heisenberg $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$'),
        ('spsm', r'$\langle S_i^+ S_j^- \rangle$'),
        ('szsz', r'$\langle S_i^z S_j^z \rangle$'),
    ]
    
    for idx, (bond_type, title) in enumerate(bond_types):
        row, col = idx // 2, idx % 2
        if bond_type in data or (bond_type == 'spsm' and 'spsm' in data):
            try:
                plot_nematic_by_orientation(data, bond_type, ax=axes[row, col], title=title)
            except Exception as e:
                print(f"Error plotting {bond_type}: {e}")
    
    # Add summary in the 4th subplot
    ax = axes[1, 1]
    ax.axis('off')
    
    if 'nematic' in data:
        summary = "NEMATIC ORDER SUMMARY\n" + "=" * 30 + "\n\n"
        summary += f"{'Bond Type':<15} {'m_nem':<10} {'Anisotropy':<12}\n"
        summary += "-" * 40 + "\n"
        
        for name, vals in data['nematic'].items():
            summary += f"{name:<15} {vals['m_nem']:<10.4f} {vals['anisotropy']:<12.4f}\n"
        
        summary += "\n" + "-" * 40 + "\n"
        summary += "ψ_nem = Σ ω^α O̅_α\n"
        summary += "where ω = exp(2πi/3)\n"
        summary += "and O̅_α = avg bond value for orientation α"
        
        ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    
    output_file = output_prefix + '_nematic_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved nematic comparison plot to: {output_file}")
    
    return fig
    
    if show:
        plt.show()
    
    return fig


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_lattice_bonds.py <order_parameters.h5> [output_prefix]")
        print("\nExample:")
        print("  python plot_lattice_bonds.py test_kagome_2x3/structure_factor_results/order_parameters.h5")
        return
    
    h5_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        output_prefix = sys.argv[2]
    else:
        output_prefix = os.path.splitext(h5_path)[0]
    
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found")
        return
    
    print(f"Loading bond data from: {h5_path}")
    data = load_bond_data(h5_path)
    
    n_sites = len(data['positions'])
    n_bonds = len(data['edges'])
    print(f"Loaded {n_sites} sites and {n_bonds} bonds")
    
    # Print bond statistics
    if 'heisenberg' in data:
        heis = data['heisenberg']
        print(f"<S·S> bonds: min={np.min(heis):.4f}, max={np.max(heis):.4f}, avg={np.mean(heis):.4f}")
    
    if 'spsm' in data:
        spsm = np.real(data['spsm'])
        print(f"<S+S-> bonds: min={np.min(spsm):.4f}, max={np.max(spsm):.4f}, avg={np.mean(spsm):.4f}")
    
    if 'szsz' in data:
        szsz = data['szsz']
        print(f"<SzSz> bonds: min={np.min(szsz):.4f}, max={np.max(szsz):.4f}, avg={np.mean(szsz):.4f}")
    
    # Print nematic order summary
    if 'nematic' in data:
        print("\nNematic order parameters:")
        for name, vals in data['nematic'].items():
            print(f"  {name:12s}: m_nem = {vals['m_nem']:.6f}, anisotropy = {vals['anisotropy']:.6f}")
    
    # Create comparison plot
    create_comparison_plot(data, output_prefix, show=False)
    
    # Create nematic comparison plot
    if 'orientation' in data:
        create_nematic_comparison_plot(data, output_prefix)
    
    # Also create individual plots
    for bond_type in ['heisenberg', 'spsm', 'szsz']:
        if bond_type in data or (bond_type == 'spsm' and 'spsm' in data):
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            try:
                plot_lattice_bonds(data, bond_type, ax=ax)
                output_file = f"{output_prefix}_{bond_type}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"Saved {bond_type} plot to: {output_file}")
                plt.close()
            except ValueError as e:
                print(f"Skipping {bond_type}: {e}")
                plt.close()


if __name__ == '__main__':
    main()
