#!/usr/bin/env python3
"""
NLCE-FTLM Convergence Analysis - Runs NLCE calculations with FTLM for increasing orders

This script performs multiple NLCE-FTLM calculations with increasing maximum orders
and plots thermodynamic properties to visualize convergence.

It builds on the NLCE-FTLM workflow in nlce_ftlm.py to:
1. Run NLCE-FTLM calculations for orders 1 to max_order
2. Collect thermodynamic data from each calculation
3. Plot properties from all orders together to visualize convergence
4. Support resummation methods for series acceleration

Key differences from nlce_convergence.py:
- Uses FTLM instead of full diagonalization
- Includes error bars from FTLM sampling
- Supports resummation methods (auto, direct, euler, wynn)
- Can handle larger cluster sizes
"""

import os
import sys
import subprocess
import argparse
import time
import glob
import logging
import numpy as np
from matplotlib.cm import get_cmap
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def run_nlce_ftlm_for_order(order, args):
    """Run NLCE-FTLM calculation for a specific maximum order"""
    logging.info("="*80)
    logging.info(f"Running NLCE-FTLM with maximum order {order}")
    logging.info("="*80)
    
    # Build command for running nlce_ftlm.py with current order
    cmd = [
        'python3', 
        'util/nlce_ftlm.py',
        f'--max_order={order}',
        f'--base_dir={args.base_dir}/order_{order}',
        f'--ed_executable={args.ed_executable}',
        f'--Jxx={args.Jxx}',
        f'--Jyy={args.Jyy}',
        f'--Jzz={args.Jzz}',
        f'--h={args.h}',
        f'--ftlm_samples={args.ftlm_samples}',
        f'--krylov_dim={args.krylov_dim}',
        f'--temp_min={args.temp_min}',
        f'--temp_max={args.temp_max}',
        f'--temp_bins={args.temp_bins}',
    ]
    
    # Add optional arguments
    if args.symmetrized:
        cmd.append('--symmetrized')
    if args.parallel:
        cmd.append('--parallel')
    if args.num_cores:
        cmd.append(f'--num_cores={args.num_cores}')
    if args.SI_units:
        cmd.append('--SI_units')
    if args.skip_cluster_gen:
        cmd.append('--skip_cluster_gen')
    if args.order_cutoff:
        cmd.append(f'--order_cutoff={args.order_cutoff}')
    if args.robust_pipeline:
        cmd.append('--robust_pipeline')
        cmd.append(f'--n_spins_per_unit={args.n_spins_per_unit}')
    
    # Add resummation method
    cmd.append(f'--resummation={args.resummation}')
        
    # Add field direction if specified
    if args.field_dir:
        cmd.extend(['--field_dir', str(args.field_dir[0]), str(args.field_dir[1]), str(args.field_dir[2])])
    
    # Run NLCE-FTLM workflow for this order
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running NLCE-FTLM for order {order}: {e}")
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
    
    # Initialize plot for each property
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Get colormap for different orders
    cmap = plt.get_cmap('viridis')
    
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
                
                # Plot this order with error bars
                color = cmap(order / max_order)
                
                if prop_error is not None and args.plot_errors:
                    ax.errorbar(temp, prop_data, yerr=prop_error, 
                               fmt='-', color=color, alpha=0.7,
                               label=f'Order {order}', linewidth=2,
                               errorevery=max(1, len(temp)//20), capsize=3)
                else:
                    ax.plot(temp, prop_data, '-', color=color, 
                           label=f'Order {order}', linewidth=2, alpha=0.8)
    
    # Add legend to each plot and adjust layout
    for ax in axes:
        ax.legend(loc='best', fontsize=9)
    
    plt.suptitle(f'NLCE-FTLM Convergence Analysis (max order {max_order})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'convergence_all_properties.png'), dpi=300)
    plt.close()
    
    logging.info(f"Combined convergence plot saved to {plot_dir}/convergence_all_properties.png")
    
    # Also create individual plots for each property for better detail
    for prop in properties:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(property_labels[prop], fontsize=14, fontweight='bold')
        ax.set_xlabel('Temperature (K)' if args.SI_units else 'Temperature', fontsize=12)
        ax.set_ylabel(property_labels[prop], fontsize=12)
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
                
                # Plot this order with error bars
                color = cmap(order / max_order)
                
                if prop_error is not None and args.plot_errors:
                    ax.errorbar(temp, prop_data, yerr=prop_error,
                               fmt='-', color=color, alpha=0.7,
                               label=f'Order {order}', linewidth=2,
                               errorevery=max(1, len(temp)//20), capsize=3)
                else:
                    ax.plot(temp, prop_data, '-', color=color,
                           label=f'Order {order}', linewidth=2, alpha=0.8)
        
        ax.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'convergence_{prop}.png'), dpi=300)
        plt.close()
        
        logging.info(f"Individual plot saved: convergence_{prop}.png")
    
    logging.info(f"All convergence plots saved to {plot_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='NLCE-FTLM convergence analysis with multiple orders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic convergence study
  python nlce_ftlm_convergence.py --max_order 4 --base_dir nlce_ftlm_conv
  
  # With custom FTLM parameters and parallel execution
  python nlce_ftlm_convergence.py --max_order 5 --ftlm_samples 50 --krylov_dim 200 \\
      --parallel --num_cores 8
  
  # With SI units and error bars
  python nlce_ftlm_convergence.py --max_order 4 --SI_units --plot_errors
  
  # Skip calculations and only plot existing results
  python nlce_ftlm_convergence.py --max_order 4 --skip_calculations
        """
    )
    
    # Parameters for the convergence study
    parser.add_argument('--max_order', type=int, required=True, 
                        help='Maximum order to calculate (will run from 1 to this value)')
    parser.add_argument('--base_dir', type=str, default='./nlce_ftlm_convergence', 
                        help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED', 
                        help='Path to the ED executable')
    
    # Model parameters
    parser.add_argument('--Jxx', type=float, default=1.0, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, default=1.0, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz coupling')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, 
                       default=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], 
                       help='Field direction (x,y,z)')
    
    # FTLM parameters
    parser.add_argument('--ftlm_samples', type=int, default=80, 
                       help='Number of random samples for FTLM')
    parser.add_argument('--krylov_dim', type=int, default=1000, 
                       help='Krylov subspace dimension for FTLM')
    parser.add_argument('--temp_min', type=float, default=0.001, 
                       help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=20.0, 
                       help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, 
                       help='Number of temperature bins')
    
    # NLCE parameters
    parser.add_argument('--order_cutoff', type=int, 
                       help='Maximum order for NLCE summation')
    parser.add_argument('--resummation', type=str, default='euler',
                       choices=['auto', 'direct', 'euler', 'wynn', 'theta', 'robust'],
                       help='Resummation method for series acceleration (default: auto)')
    
    # Control flow
    parser.add_argument('--skip_calculations', action='store_true', 
                        help='Skip calculations and only plot existing results')
    parser.add_argument('--start_order', type=int, default=1,
                        help='Starting order (default is 1)')
    parser.add_argument('--skip_cluster_gen', action='store_true',
                        help='Skip cluster generation (reuse existing clusters)')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true', 
                       help='Run FTLM in parallel')
    parser.add_argument('--num_cores', type=int, default=None, 
                       help='Number of cores to use for parallel processing')
    
    # Robust pipeline options
    parser.add_argument('--robust_pipeline', action='store_true',
                       help='Use robust two-pipeline cross-validation for C(T)')
    parser.add_argument('--n_spins_per_unit', type=int, default=4,
                       help='Spins per expansion unit (default: 4 for pyrochlore tetrahedron)')
    
    # Output options
    parser.add_argument('--symmetrized', action='store_true', 
                       help='Use symmetrized Hamiltonian')
    parser.add_argument('--SI_units', action='store_true', 
                       help='Use SI units for output')
    parser.add_argument('--plot_errors', action='store_true',
                       help='Plot error bars from FTLM sampling')
    
    # Plot limits
    parser.add_argument('--energy_ymin', type=float, default=None, 
                       help='Y-axis minimum for energy plot')
    parser.add_argument('--energy_ymax', type=float, default=None, 
                       help='Y-axis maximum for energy plot')
    parser.add_argument('--specific_heat_ymin', type=float, default=0, 
                       help='Y-axis minimum for specific heat plot')
    parser.add_argument('--specific_heat_ymax', type=float, default=0.6, 
                       help='Y-axis maximum for specific heat plot')
    parser.add_argument('--entropy_ymin', type=float, default=0, 
                       help='Y-axis minimum for entropy plot')
    parser.add_argument('--entropy_ymax', type=float, default=None, 
                       help='Y-axis maximum for entropy plot')
    parser.add_argument('--free_energy_ymin', type=float, default=None, 
                       help='Y-axis minimum for free energy plot')
    parser.add_argument('--free_energy_ymax', type=float, default=None, 
                       help='Y-axis maximum for free energy plot')
    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_ftlm_convergence.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info(f"Starting NLCE-FTLM convergence analysis up to order {args.max_order}")
    logging.info("="*80)
    logging.info(f"FTLM samples: {args.ftlm_samples}")
    logging.info(f"Krylov dimension: {args.krylov_dim}")
    logging.info(f"Temperature range: [{args.temp_min}, {args.temp_max}]")
    logging.info(f"Resummation method: {args.resummation}")
    if args.SI_units:
        logging.info("Using SI units (J/(K·mol) for C_v, S; J/mol for E, F)")
    logging.info("="*80)
    
    # Run NLCE-FTLM for each order unless skipped
    if not args.skip_calculations:
        for order in range(args.start_order, args.max_order + 1):
            success = run_nlce_ftlm_for_order(order, args)
            if not success:
                logging.warning(f"Failed to complete order {order}. Continuing with next order.")
    else:
        logging.info("Skipping calculations, using existing results.")
    
    # Collect and plot results
    collect_and_plot_results(args.max_order, args)
    
    logging.info("="*80)
    logging.info("NLCE-FTLM convergence analysis completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
