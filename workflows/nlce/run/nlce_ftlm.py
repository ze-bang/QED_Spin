#!/usr/bin/env python3
"""
NLCE Workflow with FTLM - Automates the entire NLCE calculation process using FTLM.

This script orchestrates the full Numerical Linked Cluster Expansion workflow using
the Finite Temperature Lanczos Method (FTLM) instead of full diagonalization:

1. Generate topologically distinct clusters on the pyrochlore lattice
2. Prepare Hamiltonian parameters for each cluster
3. Run FTLM for each cluster to compute thermodynamic properties directly
4. Perform NLCE summation to obtain bulk thermodynamic properties

Key differences from the full diagonalization approach:
- Uses FTLM to sample thermodynamics without computing full spectrum
- Handles larger clusters (scales to ~20-30 sites vs ~15 sites for full ED)
- Outputs include error bars from FTLM sampling
"""

import os
import sys
import glob
import re
import logging
import argparse
import subprocess
import multiprocessing
import numpy as np
from tqdm import tqdm


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


def get_cluster_files(cluster_info_dir):
    """Get list of cluster files and extract their IDs and orders"""
    cluster_files = glob.glob(os.path.join(cluster_info_dir, "cluster_*_order_*.dat"))
    clusters = []
    
    for file_path in cluster_files:
        basename = os.path.basename(file_path)
        match = re.search(r'cluster_(\d+)_order_(\d+)', basename)
        if match:
            cluster_id = int(match.group(1))
            order = int(match.group(2))
            clusters.append((cluster_id, order, file_path))
    
    return sorted(clusters)


def run_ftlm_for_cluster(args):
    """Run FTLM for a single cluster"""
    cluster_id, order, ed_executable, ham_dir, ftlm_dir, ftlm_options, symmetrized = args
    
    # Create output directory for FTLM results
    cluster_ftlm_dir = os.path.join(ftlm_dir, f'cluster_{cluster_id}_order_{order}')
    os.makedirs(cluster_ftlm_dir, exist_ok=True)
    
    # Set up FTLM command
    ham_subdir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
    
    if not os.path.exists(ham_subdir):
        logging.warning(f"Hamiltonian directory not found for cluster {cluster_id}")
        return False
    
    # Get number of sites from the site info file
    site_info_file = os.path.join(ham_subdir, "*_site_info.dat")
    site_info_files = glob.glob(site_info_file)
    
    if not site_info_files:
        logging.warning(f"Site info file not found for cluster {cluster_id}")
        return False
    
    # Count lines in site info file to get number of sites
    num_sites = 0
    with open(site_info_files[0], 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                num_sites += 1
    
    # Build FTLM command
    cmd = [
        ed_executable,
        ham_subdir,
        '--method=FTLM_GPU',
        f'--output={cluster_ftlm_dir}/output',
        f'--num_sites={num_sites}',
        '--spin_length=0.5',
        f'--samples={ftlm_options["num_samples"]}',
        f'--krylov_dim={ftlm_options["krylov_dim"]}',
    ]
    
    if symmetrized:
        cmd.append('--symmetrized')
    
    # Add thermodynamic parameters
    cmd.extend([
        '--thermo',
        f'--temp_min={ftlm_options["temp_min"]}',
        f'--temp_max={ftlm_options["temp_max"]}',
        f'--temp_bins={ftlm_options["temp_bins"]}'
    ])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"FTLM completed for cluster {cluster_id}")
        return True
        
    except subprocess.CalledProcessError as e:
        # Check if output exists despite error (crash during cleanup)
        expected_output = os.path.join(cluster_ftlm_dir, 'output', 'thermo', 'ftlm_thermo.txt')
        
        if os.path.exists(expected_output):
            logging.warning(f"FTLM for cluster {cluster_id} crashed with exit code {e.returncode} "
                          f"but output file exists - treating as success")
            return True
        
        # Log the error
        if e.returncode == -11:  # SIGSEGV
            logging.error(f"FTLM for cluster {cluster_id} crashed with SIGSEGV")
        else:
            logging.error(f"Error running FTLM for cluster {cluster_id}: {e}")
        
        if e.stdout:
            logging.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"Stderr: {e.stderr}")
        
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Automate NLCE workflow using FTLM for pyrochlore lattice',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic FTLM-based NLCE calculation
  python nlce_ftlm.py --max_order 4 --base_dir nlce_ftlm_results
  
  # With custom FTLM parameters
  python nlce_ftlm.py --max_order 4 --ftlm_samples 50 --krylov_dim 200
  
  # Parallel execution with 8 cores
  python nlce_ftlm.py --max_order 5 --parallel --num_cores 8
        """
    )
    
    # General parameters
    parser.add_argument('--max_order', type=int, required=True, 
                       help='Maximum order of clusters to generate')
    parser.add_argument('--base_dir', type=str, default='./nlce_ftlm_results', 
                       help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='../../../build/ED', 
                       help='Path to the ED executable')
    
    # Model parameters
    parser.add_argument('--Jxx', type=float, default=1.0, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, default=1.0, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz coupling')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, 
                       default=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], 
                       help='Field direction (x,y,z)')
    parser.add_argument('--random_field_width', type=float, default=0, 
                       help='Width of the random transverse field')
    
    # FTLM parameters
    parser.add_argument('--ftlm_samples', type=int, default=40, 
                       help='Number of random samples for FTLM')
    parser.add_argument('--krylov_dim', type=int, default=150, 
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
    parser.add_argument('--resummation', type=str, default='auto',
                       choices=['auto', 'direct', 'euler', 'wynn', 'theta', 'robust'],
                       help='Resummation method for series acceleration (default: auto)')
    
    # Control flow
    parser.add_argument('--skip_cluster_gen', action='store_true', 
                       help='Skip cluster generation step')
    parser.add_argument('--skip_ham_prep', action='store_true', 
                       help='Skip Hamiltonian preparation step')
    parser.add_argument('--skip_ftlm', action='store_true', 
                       help='Skip FTLM calculation step')
    parser.add_argument('--skip_nlc', action='store_true', 
                       help='Skip NLCE summation step')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true', 
                       help='Run FTLM in parallel')
    parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count(), 
                       help='Number of cores to use for parallel processing')
    
    # Robust pipeline options
    parser.add_argument('--robust_pipeline', action='store_true',
                       help='Use robust two-pipeline cross-validation for C(T)')
    parser.add_argument('--n_spins_per_unit', type=int, default=4,
                       help='Spins per expansion unit (default: 4 for pyrochlore tetrahedron)')
    
    # Other options
    parser.add_argument('--symmetrized', action='store_true', 
                       help='Use symmetrized Hamiltonian')
    parser.add_argument('--SI_units', action='store_true', 
                       help='Use SI units for output')
    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_ftlm_workflow.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info("NLCE Workflow with FTLM")
    logging.info("="*80)
    logging.info(f"Max order: {args.max_order}")
    logging.info(f"FTLM samples: {args.ftlm_samples}")
    logging.info(f"Krylov dimension: {args.krylov_dim}")
    logging.info(f"Temperature range: [{args.temp_min}, {args.temp_max}]")
    logging.info("="*80)
    
    # Define directory structure
    cluster_dir = os.path.join(args.base_dir, f'clusters_order_{args.max_order}')
    ham_dir = os.path.join(args.base_dir, f'hamiltonians_order_{args.max_order}')
    ftlm_dir = os.path.join(args.base_dir, f'ftlm_results_order_{args.max_order}')
    nlc_dir = os.path.join(args.base_dir, f'nlc_results_order_{args.max_order}')
    
    # Create directories
    for directory in [cluster_dir, ham_dir, ftlm_dir, nlc_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Generate clusters
    if not args.skip_cluster_gen:
        logging.info("="*80)
        logging.info("Step 1: Generating topologically distinct clusters")
        logging.info("="*80)
        
        cmd = [
            'python3', 
            os.path.join(os.path.dirname(__file__), '..', 'prep', 'generate_pyrochlore_clusters.py'),
            f'--max_order={args.max_order}',
            f'--output_dir={cluster_dir}',
            '--subunit=site'  # Use site-based NLCE (L(c)=0.25 for single site)
        ]
        
        logging.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logging.info("Cluster generation completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error generating clusters: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping cluster generation step.")
    
    # Get list of generated clusters
    cluster_info_dir = os.path.join(cluster_dir, f'cluster_info_order_{args.max_order}')
    if not os.path.exists(cluster_info_dir):
        logging.error(f"Cluster info directory not found: {cluster_info_dir}")
        sys.exit(1)
    
    clusters = get_cluster_files(cluster_info_dir)
    
    if not clusters:
        logging.error("No cluster files found.")
        sys.exit(1)
    
    logging.info(f"Found {len(clusters)} clusters to process.")
    
    # Step 2: Prepare Hamiltonian parameters for each cluster
    if not args.skip_ham_prep:
        logging.info("="*80)
        logging.info("Step 2: Preparing Hamiltonian parameters for each cluster")
        logging.info("="*80)
        
        for cluster_id, order, file_path in tqdm(clusters, desc="Preparing Hamiltonians"):
            logging.info(f"Preparing Hamiltonian for cluster {cluster_id} (order {order})")
            
            cluster_ham_dir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
            os.makedirs(cluster_ham_dir, exist_ok=True)
            
            # Run helper_cluster.py (now in python/edlib/)
            cmd = [
                'python3',
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python', 'edlib', 'helper_cluster.py'),
                str(args.Jxx),
                str(args.Jyy),
                str(args.Jzz),
                str(args.h),
                str(args.field_dir[0]),
                str(args.field_dir[1]),
                str(args.field_dir[2]),
                cluster_ham_dir,
                file_path,
                str(args.random_field_width),
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error preparing Hamiltonian for cluster {cluster_id}: {e}")
                if e.stdout:
                    logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
                if e.stderr:
                    logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
    else:
        logging.info("Skipping Hamiltonian preparation step.")
    
    # Step 3: Run FTLM for each cluster
    if not args.skip_ftlm:
        logging.info("="*80)
        logging.info("Step 3: Running FTLM for each cluster")
        logging.info("="*80)
        
        # Prepare FTLM options
        ftlm_options = {
            "num_samples": args.ftlm_samples,
            "krylov_dim": args.krylov_dim,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "temp_bins": args.temp_bins
        }
        
        # Prepare arguments for each cluster
        ftlm_tasks = []
        for cluster_id, order, _ in clusters:
            ftlm_tasks.append((cluster_id, order, args.ed_executable, ham_dir, 
                             ftlm_dir, ftlm_options, args.symmetrized))
        
        if args.parallel:
            logging.info(f"Running FTLM in parallel with {args.num_cores} cores")
            with multiprocessing.Pool(processes=args.num_cores) as pool:
                results = list(tqdm(
                    pool.imap(run_ftlm_for_cluster, ftlm_tasks),
                    total=len(ftlm_tasks),
                    desc="Running FTLM"
                ))
            
            success_count = sum(results)
            logging.info(f"FTLM completed for {success_count} of {len(ftlm_tasks)} clusters")
        else:
            # Run sequentially
            results = []
            for task in tqdm(ftlm_tasks, desc="Running FTLM"):
                results.append(run_ftlm_for_cluster(task))
            
            success_count = sum(results)
            logging.info(f"FTLM completed for {success_count} of {len(ftlm_tasks)} clusters")
    else:
        logging.info("Skipping FTLM calculation step.")
    
    # Step 3.5: Plot FTLM thermodynamic data for each cluster
    if not args.skip_ftlm:
        logging.info("="*80)
        logging.info("Step 3.5: Plotting FTLM thermodynamic data for each cluster")
        logging.info("="*80)
        
        # Create directory for thermodynamic plots
        thermo_plots_dir = os.path.join(args.base_dir, f'ftlm_plots_order_{args.max_order}')
        os.makedirs(thermo_plots_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Iterate through all clusters
            for cluster_id, order, _ in tqdm(clusters, desc="Plotting FTLM thermodynamic data"):
                cluster_ftlm_dir = os.path.join(ftlm_dir, f'cluster_{cluster_id}_order_{order}')
                
                # Check if FTLM thermodynamic data exists
                ftlm_thermo_file = os.path.join(cluster_ftlm_dir, "output/thermo/ftlm_thermo.txt")
                if not os.path.exists(ftlm_thermo_file):
                    logging.warning(f"No FTLM thermodynamic data found for cluster {cluster_id}")
                    continue
                
                # Load FTLM thermodynamic data
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
                        axs[0, 0].plot(T, y, 'r-', label='Energy')
                        if energy_err_idx is not None and energy_err_idx < sorted_data.shape[1]:
                            err = sorted_data[:, energy_err_idx]
                            axs[0, 0].fill_between(T, y-err, y+err, alpha=0.3, color='r')
                    axs[0, 0].set_xlabel("Temperature")
                    axs[0, 0].set_ylabel("Energy per site")
                    axs[0, 0].set_xscale('log')
                    axs[0, 0].grid(True)
                    axs[0, 0].legend()
                    
                    # Plot specific heat
                    if spec_heat_idx is not None:
                        y = sorted_data[:, spec_heat_idx]
                        axs[0, 1].plot(T, y, 'b-', label='Specific Heat')
                        if spec_heat_err_idx is not None and spec_heat_err_idx < sorted_data.shape[1]:
                            err = sorted_data[:, spec_heat_err_idx]
                            axs[0, 1].fill_between(T, y-err, y+err, alpha=0.3, color='b')
                    axs[0, 1].set_xlabel("Temperature")
                    axs[0, 1].set_ylabel("Specific Heat")
                    axs[0, 1].set_xscale('log')
                    axs[0, 1].grid(True)
                    axs[0, 1].legend()
                    
                    # Plot entropy
                    if entropy_idx is not None:
                        y = sorted_data[:, entropy_idx]
                        axs[1, 0].plot(T, y, 'g-', label='Entropy')
                        if entropy_err_idx is not None and entropy_err_idx < sorted_data.shape[1]:
                            err = sorted_data[:, entropy_err_idx]
                            axs[1, 0].fill_between(T, y-err, y+err, alpha=0.3, color='g')
                    axs[1, 0].set_xlabel("Temperature")
                    axs[1, 0].set_ylabel("Entropy per site")
                    axs[1, 0].set_xscale('log')
                    axs[1, 0].grid(True)
                    axs[1, 0].legend()
                    
                    # Plot free energy
                    if free_energy_idx is not None:
                        y = sorted_data[:, free_energy_idx]
                        axs[1, 1].plot(T, y, 'm-', label='Free Energy')
                        if free_energy_err_idx is not None and free_energy_err_idx < sorted_data.shape[1]:
                            err = sorted_data[:, free_energy_err_idx]
                            axs[1, 1].fill_between(T, y-err, y+err, alpha=0.3, color='m')
                    axs[1, 1].set_xlabel("Temperature")
                    axs[1, 1].set_ylabel("Free Energy per site")
                    axs[1, 1].set_xscale('log')
                    axs[1, 1].grid(True)
                    axs[1, 1].legend()
                    
                    # Save plot
                    plt.tight_layout()
                    plt.savefig(os.path.join(thermo_plots_dir, 
                                           f"ftlm_thermo_cluster_{cluster_id}_order_{order}.png"))
                    plt.close(fig)
                    
                    logging.info(f"FTLM thermodynamic plots created for cluster {cluster_id}")
                    
                except Exception as e:
                    logging.error(f"Error plotting FTLM data for cluster {cluster_id}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
        
        except ImportError:
            logging.error("Matplotlib not installed. Skipping FTLM thermodynamic plots.")
    
    # Step 4: Perform NLCE summation
    if not args.skip_nlc:
        logging.info("="*80)
        logging.info("Step 4: Performing NLCE summation")
        logging.info("="*80)
        
        nlc_params = [
            'python3',
            os.path.join(os.path.dirname(__file__), 'NLC_sum_ftlm.py'),
            f'--cluster_dir={cluster_info_dir}',
            f'--ftlm_dir={ftlm_dir}',
            f'--output_dir={nlc_dir}',
            '--plot',
            f'--temp_min={args.temp_min}',
            f'--temp_max={args.temp_max}',
            f'--temp_bins={args.temp_bins}',
            f'--resummation={args.resummation}',
        ]
        
        if args.SI_units:
            nlc_params.append('--SI_units')
            
        if args.order_cutoff:
            nlc_params.append(f'--order_cutoff={args.order_cutoff}')
        
        if args.robust_pipeline:
            nlc_params.append('--robust_pipeline')
            nlc_params.append(f'--n_spins_per_unit={args.n_spins_per_unit}')
        
        logging.info(f"Running command: {' '.join(nlc_params)}")
        try:
            subprocess.run(nlc_params, check=True)
            logging.info("NLCE summation completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error in NLCE summation: {e}")
            sys.exit(1)
    else:
        logging.info("Skipping NLCE summation step.")
    
    logging.info("="*80)
    logging.info("NLCE FTLM workflow completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
