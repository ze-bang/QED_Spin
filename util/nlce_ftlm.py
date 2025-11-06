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
        '--method=FTLM',
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
    parser.add_argument('--random_field_width', type=float, default=0, 
                       help='Width of the random transverse field')
    
    # FTLM parameters
    parser.add_argument('--ftlm_samples', type=int, default=20, 
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
            os.path.join(os.path.dirname(__file__), 'generate_pyrochlore_clusters.py'),
            f'--max_order={args.max_order}',
            f'--output_dir={cluster_dir}',
            f'--periodic'
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
            
            # Run helper_cluster.py
            cmd = [
                'python3',
                os.path.join(os.path.dirname(__file__), 'helper_cluster.py'),
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
        ]
        
        if args.SI_units:
            nlc_params.append('--SI_units')
            
        if args.order_cutoff:
            nlc_params.append(f'--order_cutoff={args.order_cutoff}')
        
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
