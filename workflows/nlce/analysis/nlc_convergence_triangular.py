#!/usr/bin/env python3
"""
NLCE Convergence Analysis for Triangular Lattice - Runs NLCE calculations with increasing order

This script performs multiple NLCE calculations with increasing maximum orders
and plots thermodynamic properties to visualize convergence for the triangular lattice.

It builds on the NLCE workflow in nlce_triangular.py to:
1. Run NLCE calculations for orders 1 to max_order
2. Collect thermodynamic data from each calculation
3. Plot properties from all orders together to visualize convergence
"""

import os
import sys
import subprocess
import argparse
import time
import glob
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from collections import defaultdict


def setup_logging(log_file):
    """Set up logging to file and console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def run_nlce_for_order(order, args):
    """Run triangular lattice NLCE calculation for a specific maximum order"""
    logging.info(f"="*80)
    logging.info(f"Running triangular NLCE with maximum order {order}")
    logging.info(f"="*80)
    
    # Build command for running nlce_triangular.py with current order
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nlce_script = os.path.join(script_dir, '..', 'run', 'nlce_triangular.py')
    
    cmd = [
        sys.executable,  # Use the same Python interpreter as the current script
        nlce_script,
        f'--max_order={order}',
        f'--base_dir={args.base_dir}/order_{order}',
        f'--ed_executable={args.ed_executable}',
        f'--J1={args.J1}',
        f'--J2={args.J2}',
        f'--Jz_ratio={args.Jz_ratio}',
        f'--h={args.h}',
        f'--model={args.model}',
        f'--method={args.method}',
        f'--temp_min={args.temp_min}',
        f'--temp_max={args.temp_max}',
        f'--temp_bins={args.temp_bins}',
    ]
    
    # Add field direction
    if args.field_dir:
        cmd.extend(['--field_dir', str(args.field_dir[0]), str(args.field_dir[1]), str(args.field_dir[2])])
    
    # Add anisotropic model parameters if specified
    if args.Jzz is not None:
        cmd.append(f'--Jzz={args.Jzz}')
    if args.Jpm is not None:
        cmd.append(f'--Jpm={args.Jpm}')
    if args.Jpmpm is not None:
        cmd.append(f'--Jpmpm={args.Jpmpm}')
    if args.Jzpm is not None:
        cmd.append(f'--Jzpm={args.Jzpm}')
    
    # Add optional arguments
    if args.thermo:
        cmd.append('--thermo')
    if args.parallel:
        cmd.append('--parallel')
    if args.num_cores:
        cmd.append(f'--num_cores={args.num_cores}')
    if args.SI_units:
        cmd.append('--SI_units')
    if args.skip_cluster_gen:
        cmd.append('--skip_cluster_gen')
    
    # Resummation method
    cmd.append(f'--resummation={args.resummation}')
    
    # Add automatic method selection options
    if args.no_auto_method:
        cmd.append('--no_auto_method')
    cmd.append(f'--full_ed_threshold={args.full_ed_threshold}')
    cmd.append(f'--block_size={args.block_size}')
    cmd.append(f'--symm_threshold={args.symm_threshold}')
    
    # GPU acceleration
    if args.use_gpu:
        cmd.append('--use_gpu')
    
    # Additional options
    if args.measure_spin:
        cmd.append('--measure_spin')
    if args.visualize:
        cmd.append('--visualize')
    
    # Run NLCE workflow for this order
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running triangular NLCE for order {order}: {e}")
        return False


def collect_and_plot_results(max_order, args):
    """Collect results from all orders and create comparative plots"""
    logging.info("="*80)
    logging.info("Creating convergence plots for all orders")
    logging.info("="*80)
    
    # Create directory for convergence plots
    plot_dir = os.path.join(args.base_dir, 'convergence_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Define thermodynamic properties to plot
    properties = ['energy', 'specific_heat', 'entropy', 'free_energy']
    property_files = {
        'energy': 'nlc_energy.txt',
        'specific_heat': 'nlc_specific_heat.txt',
        'entropy': 'nlc_entropy.txt',
        'free_energy': 'nlc_free_energy.txt'
    }
    
    if args.SI_units:
        property_labels = {
            'energy': 'Energy per site (J/mol)',
            'specific_heat': 'Specific Heat per site (J/(K·mol))',
            'entropy': 'Entropy per site (J/(K·mol))',
            'free_energy': 'Free Energy per site (J/mol)'
        }
    else:
        property_labels = {
            'energy': 'Energy per site',
            'specific_heat': 'Specific Heat per site',
            'entropy': 'Entropy per site',
            'free_energy': 'Free Energy per site'
        }
    
    # Get y-axis limits
    y_limits = {}
    for prop in properties:
        y_min = getattr(args, f"{prop}_ymin", None)
        y_max = getattr(args, f"{prop}_ymax", None)
        y_limits[prop] = (y_min, y_max)
    
    # Initialize plot for each property (2x2 grid for 4 properties)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get colormap for different orders
    cmap = plt.get_cmap('viridis')
    
    # Check if we should plot errors
    plot_errors = getattr(args, 'plot_errors', False)
    
    # Load data for each order and property
    data_by_order_prop = {}
    
    for order in range(1, max_order + 1):
        data_by_order_prop[order] = {}
        nlc_dir = os.path.join(args.base_dir, f'order_{order}/nlc_results_order_{order}')
        
        for prop, filename in property_files.items():
            file_path = os.path.join(nlc_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    # Load data - skip the first line (header)
                    data = np.loadtxt(file_path, comments='#')
                    data_by_order_prop[order][prop] = data
                    logging.info(f"Loaded {prop} data for order {order} from {filename}")
                except Exception as e:
                    logging.error(f"Error loading {prop} data for order {order}: {e}")
    
    # Plot each property
    for i, prop in enumerate(properties):
        ax = axes[i]
        ax.set_title(property_labels[prop], fontsize=12, fontweight='bold')
        ax.set_xlabel('Temperature (K)' if args.SI_units else 'Temperature')
        ax.set_ylabel(property_labels[prop])
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits if provided
        y_min, y_max = y_limits[prop]
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        
        for order in range(1, max_order + 1):
            if order in data_by_order_prop and prop in data_by_order_prop[order]:
                data = data_by_order_prop[order][prop]
                
                # First column is temperature, second is the property value, third is error
                temp = data[:, 0]
                prop_data = data[:, 1]
                prop_error = data[:, 2] if data.shape[1] > 2 else None
                
                # Apply temperature cutoff based on y domain if limits are provided
                if y_min is not None and y_max is not None:
                    # Find indices where data is within y limits
                    valid_indices = np.where((prop_data >= y_min) & (prop_data <= y_max))[0]
                    
                    if len(valid_indices) > 0:
                        # Find the minimum valid temperature
                        sorted_temp_indices = np.argsort(temp)
                        found_valid_temp = False
                        
                        for idx in sorted_temp_indices:
                            if idx in valid_indices:
                                min_valid_temp = temp[idx]
                                found_valid_temp = True
                                break
                        
                        if not found_valid_temp:
                            # No valid temperatures found, skip this data
                            continue
                            
                        # Filter data to only include temperatures >= min_valid_temp
                        mask = temp >= min_valid_temp
                        temp = temp[mask]
                        prop_data = prop_data[mask]
                        if prop_error is not None:
                            prop_error = prop_error[mask]
                
                # Plot this order with optional error bars
                color = cmap(order / max_order)
                
                if prop_error is not None and plot_errors:
                    ax.errorbar(temp, prop_data, yerr=prop_error, 
                               fmt='-', color=color, alpha=0.7,
                               label=f'Order {order}', linewidth=2,
                               errorevery=max(1, len(temp)//20), capsize=3)
                else:
                    ax.plot(temp, prop_data, '-', color=color, 
                            label=f'Order {order}', linewidth=2, alpha=0.8)
    
    # Add legend to each plot and adjust layout
    for ax in axes:
        ax.legend(loc='best')
    
    # Add title with model parameters
    model_info = f"Triangular Lattice: {args.model.upper()}"
    if args.model == 'anisotropic':
        model_info += f" (Jzz={args.Jzz}, Jpm={args.Jpm}, Jpmpm={args.Jpmpm}, Jzpm={args.Jzpm})"
    else:
        model_info += f" (J1={args.J1}, J2={args.J2}, Jz_ratio={args.Jz_ratio})"
    if args.h != 0:
        model_info += f", h={args.h}"
    
    fig.suptitle(model_info, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'convergence_all_properties.png'), dpi=300)
    plt.close()
    
    # Also create individual plots for each property for better detail
    for prop in properties:
        plt.figure(figsize=(10, 8))
        plt.title(f"{property_labels[prop]}\n{model_info}")
        plt.xlabel('Temperature (K)' if args.SI_units else 'Temperature')
        plt.ylabel(property_labels[prop])
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Set y-axis limits if provided
        y_min, y_max = y_limits[prop]
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        
        for order in range(1, max_order + 1):
            if order in data_by_order_prop and prop in data_by_order_prop[order]:
                data = data_by_order_prop[order][prop]
                
                # First column is temperature, second is the property value, third is error
                temp = data[:, 0]
                prop_data = data[:, 1]
                prop_error = data[:, 2] if data.shape[1] > 2 else None
                
                # Apply temperature cutoff based on y domain if limits are provided
                if y_min is not None and y_max is not None:
                    # Find indices where data is within y limits
                    valid_indices = np.where((prop_data >= y_min) & (prop_data <= y_max))[0]
                    
                    if len(valid_indices) > 0:
                        # Find the minimum valid temperature
                        sorted_temp_indices = np.argsort(temp)
                        found_valid_temp = False
                        
                        for idx in sorted_temp_indices:
                            if idx in valid_indices:
                                min_valid_temp = temp[idx]
                                found_valid_temp = True
                                break
                        
                        if not found_valid_temp:
                            # No valid temperatures found, skip this data
                            continue
                            
                        # Filter data to only include temperatures >= min_valid_temp
                        mask = temp >= min_valid_temp
                        temp = temp[mask]
                        prop_data = prop_data[mask]
                        if prop_error is not None:
                            prop_error = prop_error[mask]
                
                # Plot this order with optional error bars
                color = cmap(order / max_order)
                
                if prop_error is not None and plot_errors:
                    plt.errorbar(temp, prop_data, yerr=prop_error, 
                               fmt='-', color=color, alpha=0.7,
                               label=f'Order {order}', linewidth=2,
                               errorevery=max(1, len(temp)//20), capsize=3)
                else:
                    plt.plot(temp, prop_data, '-', color=color, 
                            label=f'Order {order}', linewidth=2, alpha=0.8)
        
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'convergence_{prop}.png'), dpi=300)
        plt.close()
    
    logging.info(f"Convergence plots saved to {plot_dir}")


def parse_cluster_file(file_path):
    """
    Parse a cluster info file to extract vertices, edges, and triangles.
    
    Returns:
        dict with 'vertices' (list of (id, x, y)), 'edges' (list of (v1, v2)),
        'triangles' (list of (v1, v2, v3)), 'order', 'multiplicity'
    """
    vertices = []
    edges = []
    triangles = []
    order = None
    multiplicity = None
    
    section = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse header info
            if line.startswith('# Order'):
                parts = line.split(':')
                if len(parts) > 1:
                    order = int(parts[1].strip().split()[0])
            elif line.startswith('# Multiplicity'):
                parts = line.split(':')
                if len(parts) > 1:
                    multiplicity = float(parts[1].strip())
            
            # Detect sections
            if '# Vertices' in line:
                section = 'vertices'
                continue
            elif '# Edges' in line:
                section = 'edges'
                continue
            elif '# Triangles' in line:
                section = 'triangles'
                continue
            elif '# Adjacency' in line:
                section = 'adjacency'
                continue
            elif '# Node Mapping' in line:
                section = 'node_mapping'
                continue
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse data based on section
            if section == 'vertices':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    vid = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    vertices.append((vid, x, y))
            elif section == 'edges':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    edges.append((int(parts[0]), int(parts[1])))
            elif section == 'triangles':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    triangles.append((int(parts[0]), int(parts[1]), int(parts[2])))
    
    return {
        'vertices': vertices,
        'edges': edges,
        'triangles': triangles,
        'order': order,
        'multiplicity': multiplicity
    }


def compute_cluster_positions_from_edges(vertices, edges):
    """
    Compute visualization positions for a cluster by building from edges.
    Uses BFS to place each site relative to its neighbors using triangular lattice vectors.
    """
    if len(vertices) == 0:
        return {}
    
    # Triangular lattice vectors (6 directions)
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    
    # All 6 neighbor directions on triangular lattice
    directions = [
        a1,        # +a1
        -a1,       # -a1
        a2,        # +a2
        -a2,       # -a2
        a1 - a2,   # +a1-a2
        -a1 + a2,  # -a1+a2
    ]
    
    # Build adjacency from edges
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    vertex_ids = [v[0] for v in vertices]
    
    # BFS to assign positions
    pos = {}
    visited = set()
    
    # Start from first node at origin
    start = vertex_ids[0]
    pos[start] = np.array([0.0, 0.0])
    visited.add(start)
    queue = [start]
    
    while queue:
        current = queue.pop(0)
        current_pos = pos[current]
        
        for neighbor in adj[current]:
            if neighbor not in visited:
                # Find direction that's consistent with existing neighbors
                best_dir = directions[0]
                visited_neighbors = [n for n in adj[neighbor] if n in visited and n != current]
                
                if visited_neighbors:
                    # Try all directions and pick one consistent with other neighbors
                    for d in directions:
                        test_pos = current_pos + d
                        consistent = True
                        for vn in visited_neighbors:
                            dist = np.linalg.norm(test_pos - pos[vn])
                            if not (0.9 < dist < 1.1):
                                consistent = False
                                break
                        if consistent:
                            best_dir = d
                            break
                
                pos[neighbor] = current_pos + best_dir
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Handle any disconnected vertices (shouldn't happen for valid clusters)
    for vid in vertex_ids:
        if vid not in pos:
            pos[vid] = np.array([0.0, 0.0])
    
    return {v: (p[0], p[1]) for v, p in pos.items()}


def visualize_single_cluster(ax, cluster_data, cluster_id):
    """
    Visualize a single cluster on a given axes.
    Site-based visualization for triangular lattice NLCE.
    Positions are computed to show proper triangular lattice geometry.
    """
    vertices = cluster_data['vertices']
    edges = cluster_data['edges']
    order = cluster_data['order']
    multiplicity = cluster_data['multiplicity']
    
    # Compute proper positions using triangular lattice geometry
    pos = compute_cluster_positions_from_edges(vertices, edges)
    
    # Center the cluster
    if pos:
        xs_all = [p[0] for p in pos.values()]
        ys_all = [p[1] for p in pos.values()]
        cx, cy = np.mean(xs_all), np.mean(ys_all)
        pos = {vid: (pos[vid][0] - cx, pos[vid][1] - cy) for vid in pos}
    
    # Draw edges
    for u, v in edges:
        if u in pos and v in pos:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                   'b-', lw=2, alpha=0.7)
    
    # Draw vertices
    xs = [pos[v][0] for v in pos]
    ys = [pos[v][1] for v in pos]
    ax.scatter(xs, ys, c='red', s=80, zorder=5, edgecolors='darkred', linewidths=1)
    
    ax.set_title(f'C{cluster_id} (n={order})\nL={multiplicity:.4f}', fontsize=9)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_clusters_for_order(order, args, plot_dir):
    """
    Create a visualization of all distinct clusters for a given order.
    """
    cluster_info_dir = os.path.join(args.base_dir, f'order_{order}/clusters_order_{order}/cluster_info_order_{order}')
    
    if not os.path.exists(cluster_info_dir):
        logging.warning(f"Cluster info directory not found: {cluster_info_dir}")
        return
    
    # Find all cluster files for this order
    cluster_files = glob.glob(os.path.join(cluster_info_dir, "cluster_*_order_*.dat"))
    
    if not cluster_files:
        logging.warning(f"No cluster files found in {cluster_info_dir}")
        return
    
    # Parse cluster files and group by order
    clusters_by_order = defaultdict(list)
    for file_path in cluster_files:
        basename = os.path.basename(file_path)
        match = re.search(r'cluster_(\d+)_order_(\d+)', basename)
        if match:
            cluster_id = int(match.group(1))
            cluster_order = int(match.group(2))
            cluster_data = parse_cluster_file(file_path)
            cluster_data['cluster_id'] = cluster_id
            clusters_by_order[cluster_order].append(cluster_data)
    
    # Sort clusters within each order by multiplicity (descending)
    for o in clusters_by_order:
        clusters_by_order[o].sort(key=lambda x: -x['multiplicity'] if x['multiplicity'] else 0)
    
    # Create figure showing all clusters
    total_clusters = sum(len(c) for c in clusters_by_order.values())
    
    if total_clusters == 0:
        return
    
    # Determine grid size
    n_cols = min(6, total_clusters)
    n_rows = (total_clusters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each cluster
    idx = 0
    for cluster_order in sorted(clusters_by_order.keys()):
        for cluster_data in clusters_by_order[cluster_order]:
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            visualize_single_cluster(ax, cluster_data, cluster_data['cluster_id'])
            idx += 1
    
    # Hide unused axes
    for i in range(idx, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    # Add title
    fig.suptitle(f'Triangular Lattice Clusters (Max Order {order})\n{total_clusters} distinct topologies', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'clusters_order_{order}.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved cluster visualization for order {order} ({total_clusters} clusters)")


def plot_all_clusters_summary(max_order, args):
    """
    Create a summary visualization showing cluster growth across all orders.
    """
    plot_dir = os.path.join(args.base_dir, 'convergence_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    logging.info("="*80)
    logging.info("Creating cluster visualizations for all orders")
    logging.info("="*80)
    
    # Plot clusters for each order
    for order in range(1, max_order + 1):
        plot_clusters_for_order(order, args, plot_dir)
    
    # Create combined summary plot showing one representative from each order
    fig, axes = plt.subplots(1, max_order, figsize=(3*max_order, 4))
    if max_order == 1:
        axes = [axes]
    
    for order in range(1, max_order + 1):
        ax = axes[order - 1]
        cluster_info_dir = os.path.join(args.base_dir, f'order_{order}/clusters_order_{order}/cluster_info_order_{order}')
        
        # Find the first cluster of this specific order (representative)
        cluster_files = glob.glob(os.path.join(cluster_info_dir, f"cluster_*_order_{order}.dat"))
        
        if cluster_files:
            # Get cluster with highest multiplicity for this order
            best_cluster = None
            best_mult = -1
            for file_path in cluster_files:
                cluster_data = parse_cluster_file(file_path)
                if cluster_data['multiplicity'] and cluster_data['multiplicity'] > best_mult:
                    best_mult = cluster_data['multiplicity']
                    best_cluster = cluster_data
                    match = re.search(r'cluster_(\d+)_order_', os.path.basename(file_path))
                    if match:
                        best_cluster['cluster_id'] = int(match.group(1))
            
            if best_cluster:
                visualize_single_cluster(ax, best_cluster, best_cluster['cluster_id'])
                ax.set_title(f'Order {order}\nL={best_mult:.4f}', fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Order {order}\n(no data)', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    fig.suptitle('Representative Clusters by Order (Triangular Lattice)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'clusters_by_order_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Cluster visualizations saved to {plot_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NLCE convergence analysis for triangular lattice with multiple orders')
    
    # Parameters for the convergence study
    parser.add_argument('--max_order', type=int, required=True, 
                        help='Maximum order to calculate (will run from 1 to this value)')
    parser.add_argument('--base_dir', type=str, default='./nlce_triangular_convergence', 
                        help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED', 
                        help='Path to the ED executable')
    
    # Model parameters for triangular lattice
    parser.add_argument('--J1', type=float, default=1.0, help='Nearest-neighbor exchange coupling')
    parser.add_argument('--J2', type=float, default=0.0, help='Next-nearest-neighbor exchange coupling')
    parser.add_argument('--Jz_ratio', type=float, default=1.0, help='Jz/Jxy ratio for XXZ model')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1], 
                       help='Field direction (x,y,z), default is out-of-plane')
    parser.add_argument('--model', type=str, default='heisenberg', 
                       choices=['heisenberg', 'xxz', 'kitaev', 'anisotropic'],
                       help='Spin model type')
    
    # Anisotropic exchange model parameters (YbMgGaO4-type)
    parser.add_argument('--Jzz', type=float, default=None, help='J_zz for anisotropic model')
    parser.add_argument('--Jpm', type=float, default=None, help='J_± for anisotropic model')
    parser.add_argument('--Jpmpm', type=float, default=None, help='J_±± for anisotropic model')
    parser.add_argument('--Jzpm', type=float, default=None, help='J_z± for anisotropic model')
    
    # ED parameters
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method (FULL, LANCZOS, etc.)')
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.1, 
                        help='Minimum temperature (default 0.1 - NLCE poorly converges at lower T for frustrated systems)')
    parser.add_argument('--temp_max', type=float, default=10.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    parser.add_argument('--resummation', type=str, default='wynn', choices=['none', 'euler', 'wynn'],
                        help='Resummation method for series acceleration (euler or wynn recommended)')
    
    # Control flow
    parser.add_argument('--skip_calculations', action='store_true', 
                        help='Skip calculations and only plot existing results')
    parser.add_argument('--start_order', type=int, default=1,
                        help='Starting order (default is 1)')
    parser.add_argument('--skip_cluster_gen', action='store_true',
                        help='Skip cluster generation and only run NLCE calculations')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true', help='Run ED in parallel')
    parser.add_argument('--num_cores', type=int, default=None, help='Number of cores to use for parallel processing')
    
    # Automatic method and symmetry selection (default behavior)
    parser.add_argument('--no_auto_method', action='store_true',
                       help='Disable automatic method selection')
    parser.add_argument('--full_ed_threshold', type=int, default=14,
                       help='Site threshold for FULL vs BLOCK_LANCZOS (default: 14)')
    parser.add_argument('--block_size', type=int, default=8,
                       help='Block size for BLOCK_LANCZOS (default: 8, should be >= degeneracy)')
    parser.add_argument('--symm_threshold', type=int, default=13,
                       help='Site threshold for using --symm flag (default: 13, only use symm for >13 sites)')
    parser.add_argument('--use_gpu', action='store_true',
                       help='Use GPU-accelerated BLOCK_LANCZOS for large clusters (requires CUDA). '
                            'Falls back to CPU if GPU is not available.')
    
    # Additional options
    parser.add_argument('--measure_spin', action='store_true',
                       help='Measure spin expectation values')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate cluster visualizations')
    
    # SI units
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')

    # Y-axis limits for plots
    parser.add_argument('--energy_ymin', type=float, default=None, help='Y-axis minimum for energy plot')
    parser.add_argument('--energy_ymax', type=float, default=None, help='Y-axis maximum for energy plot')
    parser.add_argument('--specific_heat_ymin', type=float, default=0, help='Y-axis minimum for specific heat plot')
    parser.add_argument('--specific_heat_ymax', type=float, default=None, help='Y-axis maximum for specific heat plot')
    parser.add_argument('--entropy_ymin', type=float, default=0, help='Y-axis minimum for entropy plot')
    parser.add_argument('--entropy_ymax', type=float, default=None, help='Y-axis maximum for entropy plot')
    parser.add_argument('--free_energy_ymin', type=float, default=None, help='Y-axis minimum for free energy plot')
    parser.add_argument('--free_energy_ymax', type=float, default=None, help='Y-axis maximum for free energy plot')
    
    # Plotting options
    parser.add_argument('--plot_errors', action='store_true', help='Plot error bars on convergence plots')
    parser.add_argument('--plot_clusters', action='store_true', default=True,
                       help='Generate cluster topology visualizations (default: True)')
    parser.add_argument('--no_plot_clusters', action='store_false', dest='plot_clusters',
                       help='Disable cluster topology visualizations')
    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_triangular_convergence.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info(f"Starting triangular lattice NLCE convergence analysis up to order {args.max_order}")
    logging.info(f"Model: {args.model}")
    if args.model == 'anisotropic':
        logging.info(f"Parameters: Jzz={args.Jzz}, Jpm={args.Jpm}, Jpmpm={args.Jpmpm}, Jzpm={args.Jzpm}")
    else:
        logging.info(f"Parameters: J1={args.J1}, J2={args.J2}, Jz_ratio={args.Jz_ratio}")
    logging.info(f"Field: h={args.h}, direction={args.field_dir}")
    logging.info("="*80)
    
    # Run NLCE for each order unless skipped
    if not args.skip_calculations:
        for order in range(args.start_order, args.max_order + 1):
            success = run_nlce_for_order(order, args)
            if not success:
                logging.warning(f"Failed to complete order {order}. Continuing with next order.")
    
    # Collect and plot results
    collect_and_plot_results(args.max_order, args)
    
    # Plot cluster visualizations
    if args.plot_clusters:
        plot_all_clusters_summary(args.max_order, args)
    
    logging.info("="*80)
    logging.info("Triangular lattice NLCE convergence analysis completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
