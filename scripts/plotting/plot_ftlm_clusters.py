#!/usr/bin/env python3
"""
Plot individual FTLM cluster thermodynamic data.

This script creates plots for each cluster's FTLM thermodynamic results,
showing energy, specific heat, entropy, and free energy vs temperature.
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging


def setup_logging():
    """Set up logging to console"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def get_cluster_dirs(ftlm_dir):
    """Get list of cluster directories"""
    cluster_dirs = glob.glob(os.path.join(ftlm_dir, "cluster_*_order_*"))
    clusters = []
    
    for dir_path in cluster_dirs:
        basename = os.path.basename(dir_path)
        match = re.search(r'cluster_(\d+)_order_(\d+)', basename)
        if match:
            cluster_id = int(match.group(1))
            order = int(match.group(2))
            clusters.append((cluster_id, order, dir_path))
    
    return sorted(clusters)


def plot_cluster(cluster_id, order, cluster_dir, output_dir):
    """Plot thermodynamic data for a single cluster"""
    
    # Check if FTLM thermodynamic data exists
    ftlm_thermo_file = os.path.join(cluster_dir, "output/thermo/ftlm_thermo.txt")
    if not os.path.exists(ftlm_thermo_file):
        logging.warning(f"No FTLM thermodynamic data found for cluster {cluster_id}")
        return False
    
    try:
        # Parse header to determine columns
        columns = []
        with open(ftlm_thermo_file, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    break
                stripped = line.lstrip('#').strip()
                if not stripped:
                    continue
                # Try to parse column information
                if 'Column' in stripped:
                    parts = stripped.split(':')
                    if len(parts) >= 2:
                        col_num = int(parts[0].replace('Column', '').strip()) - 1
                        col_name = parts[1].strip()
                        while len(columns) <= col_num:
                            columns.append(None)
                        columns[col_num] = col_name
                else:
                    # Try to extract column names directly
                    tokens = stripped.split()
                    if tokens:
                        columns = tokens
        
        if not columns:
            # Default expected columns for FTLM output
            columns = ['Temperature', 'Energy', 'Energy_Error', 'Specific_Heat', 
                     'Specific_Heat_Error', 'Entropy', 'Entropy_Error', 
                     'Free_Energy', 'Free_Energy_Error']
        
        # Load the data
        data = np.loadtxt(ftlm_thermo_file, comments='#')
        data = np.atleast_2d(data)
        
        # Find column indices (case-insensitive matching)
        def find_col(names):
            norm_cols = [c.lower().replace('_', ' ') if c else '' for c in columns]
            for name in names:
                norm_name = name.lower().replace('_', ' ')
                if norm_name in norm_cols:
                    return norm_cols.index(norm_name)
            return None
        
        temp_idx = find_col(['temperature', 'temp', 't'])
        if temp_idx is None:
            temp_idx = 0
        
        energy_idx = find_col(['energy', 'internal energy'])
        energy_err_idx = find_col(['energy error', 'energy err'])
        
        spec_heat_idx = find_col(['specific heat', 'specificheat', 'c'])
        spec_heat_err_idx = find_col(['specific heat error', 'specific heat err'])
        
        entropy_idx = find_col(['entropy', 's'])
        entropy_err_idx = find_col(['entropy error', 'entropy err'])
        
        free_energy_idx = find_col(['free energy', 'freeenergy', 'f'])
        free_energy_err_idx = find_col(['free energy error', 'free energy err'])
        
        # Extract temperature
        T = data[:, temp_idx]
        
        # Sort by temperature
        sort_idx = np.argsort(T)
        T = T[sort_idx]
        sorted_data = data[sort_idx]
        
        # Create plots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"FTLM Thermodynamic Properties for Cluster {cluster_id} (Order {order})")
        
        # Plot energy
        if energy_idx is not None:
            y = sorted_data[:, energy_idx]
            axs[0, 0].plot(T, y, 'r-', label='Energy', linewidth=2)
            if energy_err_idx is not None and energy_err_idx < sorted_data.shape[1]:
                err = sorted_data[:, energy_err_idx]
                axs[0, 0].fill_between(T, y-err, y+err, alpha=0.3, color='r', label='±1σ error')
        axs[0, 0].set_xlabel("Temperature", fontsize=12)
        axs[0, 0].set_ylabel("Energy per site", fontsize=12)
        axs[0, 0].set_xscale('log')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()
        
        # Plot specific heat
        if spec_heat_idx is not None:
            y = sorted_data[:, spec_heat_idx]
            axs[0, 1].plot(T, y, 'b-', label='Specific Heat', linewidth=2)
            if spec_heat_err_idx is not None and spec_heat_err_idx < sorted_data.shape[1]:
                err = sorted_data[:, spec_heat_err_idx]
                axs[0, 1].fill_between(T, y-err, y+err, alpha=0.3, color='b', label='±1σ error')
        axs[0, 1].set_xlabel("Temperature", fontsize=12)
        axs[0, 1].set_ylabel("Specific Heat", fontsize=12)
        axs[0, 1].set_xscale('log')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].legend()
        
        # Plot entropy
        if entropy_idx is not None:
            y = sorted_data[:, entropy_idx]
            axs[1, 0].plot(T, y, 'g-', label='Entropy', linewidth=2)
            if entropy_err_idx is not None and entropy_err_idx < sorted_data.shape[1]:
                err = sorted_data[:, entropy_err_idx]
                axs[1, 0].fill_between(T, y-err, y+err, alpha=0.3, color='g', label='±1σ error')
        axs[1, 0].set_xlabel("Temperature", fontsize=12)
        axs[1, 0].set_ylabel("Entropy per site", fontsize=12)
        axs[1, 0].set_xscale('log')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].legend()
        
        # Plot free energy
        if free_energy_idx is not None:
            y = sorted_data[:, free_energy_idx]
            axs[1, 1].plot(T, y, 'm-', label='Free Energy', linewidth=2)
            if free_energy_err_idx is not None and free_energy_err_idx < sorted_data.shape[1]:
                err = sorted_data[:, free_energy_err_idx]
                axs[1, 1].fill_between(T, y-err, y+err, alpha=0.3, color='m', label='±1σ error')
        axs[1, 1].set_xlabel("Temperature", fontsize=12)
        axs[1, 1].set_ylabel("Free Energy per site", fontsize=12)
        axs[1, 1].set_xscale('log')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].legend()
        
        # Save plot
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"ftlm_thermo_cluster_{cluster_id}_order_{order}.png")
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        logging.error(f"Error plotting FTLM data for cluster {cluster_id}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Plot individual FTLM cluster thermodynamic data',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--ftlm_dir', type=str, required=True,
                       help='Directory containing FTLM results for all clusters')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: ftlm_plots in parent of ftlm_dir)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Determine output directory
    if args.output_dir is None:
        parent_dir = os.path.dirname(args.ftlm_dir)
        # Extract order from ftlm_dir name
        match = re.search(r'order_(\d+)', args.ftlm_dir)
        if match:
            order = match.group(1)
            args.output_dir = os.path.join(parent_dir, f'ftlm_plots_order_{order}')
        else:
            args.output_dir = os.path.join(parent_dir, 'ftlm_plots')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("="*80)
    logging.info("Plotting FTLM cluster thermodynamic data")
    logging.info("="*80)
    logging.info(f"FTLM results directory: {args.ftlm_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    
    # Get list of clusters
    clusters = get_cluster_dirs(args.ftlm_dir)
    
    if not clusters:
        logging.error("No cluster directories found!")
        sys.exit(1)
    
    logging.info(f"Found {len(clusters)} clusters to plot")
    logging.info("="*80)
    
    # Plot each cluster
    success_count = 0
    for cluster_id, order, cluster_dir in tqdm(clusters, desc="Plotting clusters"):
        if plot_cluster(cluster_id, order, cluster_dir, args.output_dir):
            success_count += 1
            logging.info(f"Successfully plotted cluster {cluster_id} (order {order})")
    
    logging.info("="*80)
    logging.info(f"Completed: {success_count}/{len(clusters)} clusters plotted successfully")
    logging.info(f"Plots saved to: {args.output_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
