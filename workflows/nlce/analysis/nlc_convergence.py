#!/usr/bin/env python3
"""
NLCE Convergence Analysis - Runs NLCE calculations with increasing order

This script performs multiple NLCE calculations with increasing maximum orders
and plots thermodynamic properties to visualize convergence.

It builds on the NLCE workflow in nlce.py to:
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


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
    """Run NLCE calculation for a specific maximum order"""
    logging.info(f"="*80)
    logging.info(f"Running NLCE with maximum order {order}")
    logging.info(f"="*80)
    
    # Build command for running nlce.py with current order
    # Path is relative to the project root (workflows/nlce/run/nlce.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nlce_script = os.path.join(script_dir, '..', 'run', 'nlce.py')
    
    cmd = [
        sys.executable,  # Use the same Python interpreter as the current script
        nlce_script,
        f'--max_order={order}',
        f'--base_dir={args.base_dir}/order_{order}',
        f'--ed_executable={args.ed_executable}',
        f'--Jxx={args.Jxx}',
        f'--Jyy={args.Jyy}',
        f'--Jzz={args.Jzz}',
        f'--h={args.h}',
        f'--method={args.method}',
        f'--temp_min={args.temp_min}',
        f'--temp_max={args.temp_max}',
        f'--temp_bins={args.temp_bins}',
    ]
    
    # Add optional symmetrized flag (legacy)
    if args.symmetrized:
        cmd.append('--symmetrized')
    
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
    
    # Add automatic method selection options
    if args.no_auto_method:
        cmd.append('--no_auto_method')
    cmd.append(f'--full_ed_threshold={args.full_ed_threshold}')
    cmd.append(f'--block_size={args.block_size}')
    
    # GPU acceleration
    if args.use_gpu:
        cmd.append('--use_gpu')
    
    # Lanczos-boosted NLCE options
    if args.lanczos_boost:
        cmd.append('--lanczos_boost')
        cmd.append(f'--lb_site_threshold={args.lb_site_threshold}')
        cmd.append(f'--lb_n_eigenvalues={args.lb_n_eigenvalues}')
        if args.lb_energy_window is not None:
            cmd.append(f'--lb_energy_window={args.lb_energy_window}')
        if args.lb_check_convergence:
            cmd.append('--lb_check_convergence')
    
    # Additional options
    if args.measure_spin:
        cmd.append('--measure_spin')
    if args.random_field_width != 0:
        cmd.append(f'--random_field_width={args.random_field_width}')
        
    # Add field direction if specified
    if args.field_dir:
        cmd.extend(['--field_dir', str(args.field_dir[0]), str(args.field_dir[1]), str(args.field_dir[2])])
    
    # Run NLCE workflow for this order
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running NLCE for order {order}: {e}")
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
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'convergence_all_properties.png'), dpi=300)
    plt.close()
    
    # Also create individual plots for each property for better detail
    for prop in properties:
        plt.figure(figsize=(10, 8))
        plt.title(property_labels[prop])
        plt.xlabel('Temperature')
        plt.ylabel(property_labels[prop])
        plt.xscale('log')
        plt.grid(True)
        
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NLCE convergence analysis with multiple orders')
    
    # Parameters for the convergence study
    parser.add_argument('--max_order', type=int, required=True, 
                        help='Maximum order to calculate (will run from 1 to this value)')
    parser.add_argument('--base_dir', type=str, default='./nlce_convergence', 
                        help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED', 
                        help='Path to the ED executable')
    
    # Model parameters (same as nlce.py)
    parser.add_argument('--Jxx', type=float, default=1.0, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, default=1.0, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz coupling')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[1, 1, 1], help='Field direction (x,y,z)')
    
    # ED parameters (same as nlce.py)
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method (FULL, LANCZOS, etc.)')
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.001, help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    
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
    
    # Symmetrization
    parser.add_argument('--symmetrized', action='store_true', 
                       help='Legacy: force symmetrized mode (now handled automatically)')
    
    # ScaLAPACK distributed diagonalization for large clusters
    parser.add_argument('--scalapack_threshold', type=int, default=16,
                       help='Site threshold for switching to ScaLAPACK (default: 16). '
                            'Clusters with >= sites use SCALAPACK_MIXED for distributed diagonalization.')
    parser.add_argument('--no_scalapack', action='store_true',
                       help='Disable ScaLAPACK - always use standard FULL diagonalization.')
    
    # Legacy arguments kept for backwards compatibility
    parser.add_argument('--no_auto_method', action='store_true',
                       help='(Ignored) Legacy argument.')
    parser.add_argument('--full_ed_threshold', type=int, default=12,
                       help='(Ignored) Legacy argument - use --scalapack_threshold instead.')
    parser.add_argument('--block_size', type=int, default=8,
                       help='(Ignored) Legacy argument.')
    parser.add_argument('--use_gpu', action='store_true',
                       help='(Ignored) Legacy argument.')
    
    # Lanczos-Boosted NLCE Parameters
    parser.add_argument('--lanczos_boost', action='store_true',
                       help='Enable Lanczos-boosted NLCE mode. Large clusters use partial '
                            'Lanczos diagonalization (low-energy eigenstates only).')
    parser.add_argument('--lb_site_threshold', type=int, default=12,
                       help='Site threshold for LB-NLCE (default: 12)')
    parser.add_argument('--lb_n_eigenvalues', type=int, default=200,
                       help='Number of low-lying eigenvalues to compute for large clusters '
                            'in LB-NLCE mode (default: 200)')
    parser.add_argument('--lb_energy_window', type=float, default=None,
                       help='Energy window above ground state for LB-NLCE')
    parser.add_argument('--lb_check_convergence', action='store_true',
                       help='Check LB-NLCE convergence with increasing eigenvalues')
    
    # Additional options
    parser.add_argument('--measure_spin', action='store_true',
                       help='Measure spin expectation values')
    parser.add_argument('--random_field_width', type=float, default=0,
                       help='Width of the random transverse field')
    
    # SI units
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')

    # Parse arguments
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
    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_convergence.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info(f"Starting NLCE convergence analysis up to order {args.max_order}")
    logging.info("="*80)
    
    # Run NLCE for each order unless skipped
    if not args.skip_calculations:
        for order in range(args.start_order, args.max_order + 1):
            success = run_nlce_for_order(order, args)
            if not success:
                logging.warning(f"Failed to complete order {order}. Continuing with next order.")
    
    # Collect and plot results
    collect_and_plot_results(args.max_order, args)
    
    logging.info("="*80)
    logging.info("NLCE convergence analysis completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")