#!/usr/bin/env python3
"""
NLCE Workflow - Automates the entire NLCE calculation process for triangular lattice.

This script orchestrates the full Numerical Linked Cluster Expansion workflow:
1. Generate topologically distinct clusters on the triangular lattice
2. Prepare Hamiltonian parameters for each cluster (J1-J2 Heisenberg, XXZ, etc.)
3. Run Exact Diagonalization for each cluster to compute spectrum
4. Perform NLCE summation to obtain thermodynamic properties

Supports two NLCE expansion schemes:
- Site-based NLCE (default): Order = number of sites
- Triangle-based NLCE (--triangle_based): Order = number of triangles
  This gives far fewer clusters at each order, useful for frustrated systems
"""

import os
import sys
import subprocess
import argparse
import time
import glob
import re
import multiprocessing
import logging
from tqdm import tqdm
import numpy as np
import traceback

# Compute workspace root from script location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
_DEFAULT_ED_PATH = os.path.join(_WORKSPACE_ROOT, 'build', 'ED')

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def check_gpu_available():
    """Check if GPU is available for CUDA operations."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5
        )
        gpu_names = result.stdout.strip().split("\n")
        if gpu_names and gpu_names[0]:
            return True
        return False
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def setup_logging(log_file):
    """Set up logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def get_cluster_files(cluster_info_dir):
    """Get list of cluster files and extract their IDs and orders."""
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


def get_num_sites(file_path):
    """Extract number of sites from a cluster file."""
    with open(file_path, 'r') as f:
        for line in f:
            if "Number of vertices:" in line:
                return int(line.split(":")[1].strip())
    return None


def run_ed_for_cluster(args):
    """Run ED for a single cluster.
    
    Method selection based on system size:
    - FULL: For small clusters (< scalapack_threshold sites)
    - SCALAPACK_MIXED: For large clusters (distributed diagonalization with mixed precision)
    
    Uses --symm flag for automatic symmetry selection on larger clusters.
    """
    cluster_id, order, ed_executable, ham_dir, ed_dir, ed_options, use_gpu = args
    
    # Create output directory for ED results (including the 'output' subdirectory)
    cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
    output_subdir = os.path.join(cluster_ed_dir, 'output')
    os.makedirs(output_subdir, exist_ok=True)
    
    # Set up ED command
    ham_subdir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
    
    if not os.path.exists(ham_subdir):
        logging.warning(f"Hamiltonian directory not found for cluster {cluster_id}")
        return False
    
    # Get number of sites from the input file
    site_info_file = os.path.join(ham_subdir, f"*_site_info.dat")
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
    
    # Calculate Hilbert space dimension
    hilbert_dim = 2 ** num_sites
    
    # Symmetry threshold for using --symm flag
    symm_threshold = ed_options.get("symm_threshold", 13)
    use_symm = (num_sites > symm_threshold)
    
    # Threshold for switching to ScaLAPACK (distributed diagonalization)
    scalapack_threshold = ed_options.get("scalapack_threshold", 16)
    use_scalapack = (num_sites >= scalapack_threshold and ed_options.get("use_scalapack", True))
    
    symm_indicator = ' with --symm' if use_symm else ''
    
    if use_scalapack:
        # Large cluster: use ScaLAPACK with mixed precision for efficient distributed diagonalization
        logging.info(f"Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): Using SCALAPACK_MIXED{symm_indicator}")
        cmd = [
            ed_executable,
            ham_subdir,
            '--method=SCALAPACK_MIXED',
            '--eigenvalues=FULL',
            f'--output={cluster_ed_dir}/output',
            f'--num_sites={num_sites}',
            '--spin_length=0.5',
        ]
    else:
        # Small cluster: use standard FULL diagonalization
        logging.info(f"Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): Using FULL diagonalization{symm_indicator}")
        cmd = [
            ed_executable,
            ham_subdir,
            '--method=FULL',
            '--eigenvalues=FULL',
            f'--output={cluster_ed_dir}/output',
            f'--num_sites={num_sites}',
            '--spin_length=0.5',
        ]
    
    if use_symm:
        cmd.append('--symm')

    if ed_options.get("measure_spin", False):
        cmd.append('--measure_spin')
    
    # Add thermodynamic parameters if required
    if ed_options.get("thermo", False):
        cmd.extend([
            '--thermo',
            f'--temp_min={ed_options["temp_min"]}',
            f'--temp_max={ed_options["temp_max"]}',
            f'--temp_bins={ed_options["temp_bins"]}'
        ])
    
    # Set ED_PYTHON environment variable
    env = os.environ.copy()
    env['ED_PYTHON'] = sys.executable
    
    # For small matrices (num_sites <= 8), use single-threaded OpenMP to avoid race conditions
    # This is a workaround for an OpenMP bug with very small matrices
    if num_sites <= 8:
        env['OMP_NUM_THREADS'] = '1'
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        # Check if the computation actually succeeded despite the error
        expected_output_dir = os.path.join(cluster_ed_dir, 'output')
        
        if os.path.exists(expected_output_dir):
            h5_file = os.path.join(expected_output_dir, 'ed_results.h5')
            if os.path.exists(h5_file):
                logging.warning(f"ED for cluster {cluster_id} crashed but HDF5 output exists - treating as success")
                return True
        
        if e.returncode == -11:
            logging.error(f"ED for cluster {cluster_id} crashed with SIGSEGV")
        else:
            logging.error(f"Error running ED for cluster {cluster_id}: {e}")
        
        logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Automate NLCE workflow for triangular lattice')
    
    # Parameters for the entire workflow
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--base_dir', type=str, default='./nlce_triangular_results', help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default=_DEFAULT_ED_PATH, 
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
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method')
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.1, help='Minimum temperature (default 0.1 - NLCE poorly converges at lower T for frustrated systems)')
    parser.add_argument('--temp_max', type=float, default=10.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    parser.add_argument('--resummation', type=str, default='euler', choices=['none', 'euler', 'wynn'],
                       help='Resummation method for series acceleration (euler or wynn recommended)')
    
    # NLCE parameters
    parser.add_argument('--order_cutoff', type=int, help='Maximum order for NLCE summation')
    
    # Control flow
    parser.add_argument('--skip_cluster_gen', action='store_true', help='Skip cluster generation step')
    parser.add_argument('--skip_ham_prep', action='store_true', help='Skip Hamiltonian preparation step')
    parser.add_argument('--skip_ed', action='store_true', help='Skip Exact Diagonalization step')
    parser.add_argument('--skip_nlc', action='store_true', help='Skip NLCE summation step')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true', help='Run ED in parallel')
    parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count(), 
                       help='Number of cores to use for parallel processing')
    
    # SI units
    parser.add_argument('--SI_units', action='store_true', 
                       help='Convert to SI units: specific heat in J/(mol·K). '
                            'Temperature remains in units of J unless --J_kelvin is set.')
    parser.add_argument('--J_kelvin', type=float, default=None,
                       help='Exchange coupling J in Kelvin. If set with --SI_units, '
                            'temperatures are converted to Kelvin: T_K = T × J_kelvin. '
                            'Required for direct comparison with experimental data in Kelvin.')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')

    # ScaLAPACK distributed diagonalization for large clusters
    parser.add_argument('--scalapack_threshold', type=int, default=16,
                       help='Site threshold for switching to ScaLAPACK (default: 16). '
                            'Clusters with >= sites use SCALAPACK_MIXED for distributed diagonalization.')
    parser.add_argument('--no_scalapack', action='store_true',
                       help='Disable ScaLAPACK - always use standard FULL diagonalization.')
    parser.add_argument('--symm_threshold', type=int, default=13,
                       help='Site threshold for using --symm flag (default: 13)')
    
    # Legacy arguments kept for backwards compatibility
    parser.add_argument('--no_auto_method', action='store_true',
                       help='(Ignored) Legacy argument.')
    parser.add_argument('--full_ed_threshold', type=int, default=14,
                       help='(Ignored) Legacy argument - use --scalapack_threshold instead.')
    parser.add_argument('--block_size', type=int, default=8,
                       help='(Ignored) Legacy argument.')
    parser.add_argument('--use_gpu', action='store_true',
                       help='(Ignored) Legacy argument.')
    
    parser.add_argument('--visualize', action='store_true', help='Generate cluster visualizations')
    
    # NLCE expansion type (triangle-based is the default)
    parser.add_argument('--site_based', action='store_true', 
                       help='Use site-based NLCE (order = number of sites). '
                            'Default is triangle-based which gives fewer clusters.')

    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_triangular_workflow.log')
    setup_logging(log_file)
    
    # Log expansion type (triangle-based is the default)
    if args.site_based:
        logging.info("Using SITE-BASED NLCE (order = number of sites)")
    else:
        logging.info("Using TRIANGLE-BASED NLCE (order = number of triangles)")
        logging.info("  Reference cluster counts: 1,1,3,5,12,35,98,299,... (OEIS A007854)")
    
    # Define directory structure
    cluster_dir = os.path.join(args.base_dir, f'clusters_order_{args.max_order}')
    ham_dir = os.path.join(args.base_dir, f'hamiltonians_order_{args.max_order}')
    ed_dir = os.path.join(args.base_dir, f'ed_results_order_{args.max_order}')
    nlc_dir = os.path.join(args.base_dir, f'nlc_results_order_{args.max_order}')
    
    # Create directories
    for directory in [cluster_dir, ham_dir, ed_dir, nlc_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Generate clusters
    if not args.skip_cluster_gen:
        logging.info("="*80)
        if args.site_based:
            logging.info("Step 1: Generating site-based NLCE clusters (order = sites)")
        else:
            logging.info("Step 1: Generating triangle-based NLCE clusters (order = triangles)")
        logging.info("="*80)
        
        if args.site_based:
            # Use site-based cluster generator
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'prep', 'generate_triangular_clusters.py'),
                f'--max_order={args.max_order}',
                f'--output_dir={cluster_dir}',
            ]
            
            if args.visualize:
                cmd.append('--visualize')
        else:
            # Use triangle-based cluster generator (default)
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', 'prep', 'generate_triangle_nlce_clusters.py'),
                f'--max_order={args.max_order}',
                f'--output_dir={cluster_dir}',
            ]
            
            if args.visualize:
                cmd.append('--visualize')
            else:
                cmd.append('--no_visualize')
        
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
        if args.model == 'anisotropic':
            logging.info(f"Model: {args.model}, Jzz={args.Jzz}, Jpm={args.Jpm}, Jpmpm={args.Jpmpm}, Jzpm={args.Jzpm}, h={args.h}")
        else:
            logging.info(f"Model: {args.model}, J1={args.J1}, J2={args.J2}, h={args.h}")
        logging.info("="*80)
        
        for cluster_id, order, file_path in tqdm(clusters, desc="Preparing Hamiltonians"):
            logging.debug(f"Preparing Hamiltonian for cluster {cluster_id} (order {order})")
            
            # Create output directory for this cluster
            cluster_ham_dir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
            os.makedirs(cluster_ham_dir, exist_ok=True)
            
            # Run helper_cluster_triangular.py with argparse interface
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'python', 'edlib', 'helper_cluster_triangular.py'),
                '--J1', str(args.J1),
                '--J2', str(args.J2),
                '--h', str(args.h),
                '--field_dir', str(args.field_dir[0]), str(args.field_dir[1]), str(args.field_dir[2]),
                '--output_dir', cluster_ham_dir,
                '--cluster_file', file_path,
                '--model', args.model,
                '--Jz_ratio', str(args.Jz_ratio),
            ]
            
            # Add anisotropic model parameters if specified
            if args.Jzz is not None:
                cmd.extend(['--Jzz', str(args.Jzz)])
            if args.Jpm is not None:
                cmd.extend(['--Jpm', str(args.Jpm)])
            if args.Jpmpm is not None:
                cmd.extend(['--Jpmpm', str(args.Jpmpm)])
            if args.Jzpm is not None:
                cmd.extend(['--Jzpm', str(args.Jzpm)])
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error preparing Hamiltonian for cluster {cluster_id}: {e}")
                logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
                logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
    else:
        logging.info("Skipping Hamiltonian preparation step.")
    
    # Step 3: Run Exact Diagonalization for each cluster
    if not args.skip_ed:
        logging.info("="*80)
        logging.info("Step 3: Running Exact Diagonalization for each cluster")
        logging.info("="*80)
        
        ed_options = {
            "method": args.method,
            "thermo": args.thermo,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "temp_bins": args.temp_bins,
            "measure_spin": args.measure_spin,
            "symm_threshold": args.symm_threshold,
            "scalapack_threshold": args.scalapack_threshold,
            "use_scalapack": not args.no_scalapack,
        }
        
        use_gpu = False  # GPU not used for FULL/ScaLAPACK diagonalization
        
        logging.info(f"NLCE ED Configuration:")
        if not args.no_scalapack:
            logging.info(f"  - Small clusters (< {args.scalapack_threshold} sites): FULL diagonalization")
            logging.info(f"  - Large clusters (>= {args.scalapack_threshold} sites): SCALAPACK_MIXED (distributed)")
        else:
            logging.info(f"  - Method: FULL diagonalization (ScaLAPACK disabled)")
        logging.info(f"  - Symmetry: --symm for clusters with > {args.symm_threshold} sites")
        
        # Prepare arguments for each cluster
        ed_tasks = []
        for cluster_id, order, _ in clusters:
            ed_tasks.append((cluster_id, order, args.ed_executable, ham_dir, ed_dir, ed_options, use_gpu))
        
        if args.parallel:
            logging.info(f"Running ED in parallel with {args.num_cores} cores")
            with multiprocessing.Pool(processes=args.num_cores) as pool:
                results = list(tqdm(
                    pool.imap(run_ed_for_cluster, ed_tasks),
                    total=len(ed_tasks),
                    desc="Running ED"
                ))
            
            success_count = sum(results)
            logging.info(f"ED completed for {success_count} of {len(ed_tasks)} clusters")
        else:
            for task in tqdm(ed_tasks, desc="Running ED"):
                run_ed_for_cluster(task)
    else:
        logging.info("Skipping Exact Diagonalization step.")
    
    # Step 4: NLC Summation
    if not args.skip_nlc:
        logging.info("="*80)
        logging.info("Step 4: Performing NLCE Summation")
        logging.info("="*80)
        
        order_cutoff = args.order_cutoff if args.order_cutoff else args.max_order
        
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), 'NLC_sum_triangular.py'),
            f'--cluster_dir={cluster_info_dir}',
            f'--eigenvalue_dir={ed_dir}',
            f'--output_dir={nlc_dir}',
            f'--temp_min={args.temp_min}',
            f'--temp_max={args.temp_max}',
            f'--temp_bins={args.temp_bins}',
            f'--max_order={order_cutoff}',
            f'--resummation={args.resummation}',
        ]
        
        if args.measure_spin:
            cmd.append('--measure_spin')
        
        if args.SI_units:
            cmd.append('--SI_units')
        
        if args.J_kelvin is not None:
            cmd.append(f'--J_kelvin={args.J_kelvin}')
        
        logging.info(f"Running command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logging.info("NLCE summation completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running NLCE summation: {e}")
    else:
        logging.info("Skipping NLCE summation step.")
    
    logging.info("="*80)
    logging.info("NLCE workflow completed!")
    logging.info(f"Results saved to: {args.base_dir}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
