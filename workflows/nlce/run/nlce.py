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

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

#!/usr/bin/env python3
"""
NLCE Workflow - Automates the entire NLCE calculation process for pyrochlore lattice.

This script orchestrates the full Numerical Linked Cluster Expansion workflow:
1. Generate topologically distinct clusters on the pyrochlore lattice
2. Prepare Hamiltonian parameters for each cluster
3. Run Exact Diagonalization for each cluster to compute spectrum
4. Perform NLCE summation to obtain thermodynamic properties
"""


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

def get_num_sites(file_path):
    """Extract number of sites from a cluster file"""
    with open(file_path, 'r') as f:
        for line in f:
            if "Number of vertices:" in line:
                return int(line.split(":")[1].strip())
    return None

def run_ed_for_cluster(args):
    """Run ED for a single cluster
    
    Uses --symm flag by default for automatic symmetry selection:
    - Auto-selects between symmetrized and streaming-symmetry modes
    - Exploits spatial symmetries to reduce Hilbert space dimension
    
    Method selection (automatic by default):
    - FULL: For small clusters (dim <= 4096, i.e. <= 12 sites)
    - BLOCK_LANCZOS: For larger clusters with degeneracies (Heisenberg models)
    
    Block size for BLOCK_LANCZOS should be at least as large as the expected
    degeneracy to properly resolve degenerate eigenspaces.
    """
    cluster_id, order, ed_executable, ham_dir, ed_dir, ed_options, symmetrized = args
    
    # Create output directory for ED results
    cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
    os.makedirs(cluster_ed_dir, exist_ok=True)
    
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
    
    # Count lines in site info file to get number of sites (excluding header lines)
    num_sites = 0
    with open(site_info_files[0], 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                num_sites += 1
    
    # Calculate Hilbert space dimension for method selection
    hilbert_dim = 2 ** num_sites
    
    # Thresholds for method selection
    full_ed_threshold = ed_options.get("full_ed_threshold", 12)  # Use FULL for <= 12 sites (dim <= 4096)
    block_lanczos_block_size = ed_options.get("block_size", 8)  # Block size >= expected degeneracy
    
    # Determine method: FULL for small clusters, BLOCK_LANCZOS for larger ones
    # For thermodynamics, we need ALL eigenvalues, so we use --symm which auto-selects
    # the best symmetry-exploiting mode based on system size
    use_block_lanczos = (num_sites > full_ed_threshold and ed_options.get("auto_method", True))
    
    if ed_options["method"] == 'mTPQ':
        cmd = [
            ed_executable,
            ham_subdir,
            f'--method={ed_options["method"]}',
            f'--output={cluster_ed_dir}/output',
            f'--num_sites={num_sites}',
            '--spin_length=0.5',
            '--iterations=100000',
            '--large_value=100'
        ]
    elif use_block_lanczos:
        # Large cluster: use BLOCK_LANCZOS for efficient full spectrum calculation
        # Block size should be >= degeneracy (for Heisenberg, use at least 4-8)
        logging.info(f"Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): "
                    f"Using BLOCK_LANCZOS with --symm (block_size={block_lanczos_block_size})")
        cmd = [
            ed_executable,
            ham_subdir,
            '--method=BLOCK_LANCZOS',
            '--eigenvalues=FULL',
            f'--block_size={block_lanczos_block_size}',
            f'--output={cluster_ed_dir}/output',
            f'--num_sites={num_sites}',
            '--spin_length=0.5',
            '--symm',  # Auto-select best symmetry mode
        ]
    else:
        # Small cluster: use FULL diagonalization with --symm
        logging.info(f"Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): "
                    f"Using FULL with --symm")
        cmd = [
            ed_executable,
            ham_subdir,
            '--method=FULL',
            '--eigenvalues=FULL',
            f'--output={cluster_ed_dir}/output',
            f'--num_sites={num_sites}',
            '--spin_length=0.5',
            '--symm',  # Auto-select best symmetry mode
        ]

    if ed_options["measure_spin"]:
        cmd.append('--measure_spin')

    # If user explicitly requests --symmetrized, replace --symm with --symmetrized
    if symmetrized:
        # Remove --symm and add --symmetrized instead
        if '--symm' in cmd:
            cmd.remove('--symm')
        cmd.append('--symmetrized')
    
    # Add thermodynamic parameters if required
    if ed_options["thermo"]:
        cmd.extend([
            '--thermo',
            f'--temp_min={ed_options["temp_min"]}',
            f'--temp_max={ed_options["temp_max"]}',
            f'--temp_bins={ed_options["temp_bins"]}'
        ])
    
    # Set ED_PYTHON environment variable to use the same Python interpreter
    # This ensures pynauty and other dependencies are found
    env = os.environ.copy()
    env['ED_PYTHON'] = sys.executable
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        # Check if the computation actually succeeded despite the error
        # This can happen when the ED program completes successfully but crashes during cleanup
        expected_output_dir = os.path.join(cluster_ed_dir, 'output')
        
        # Check if output directory exists and has content
        if os.path.exists(expected_output_dir):
            output_files = os.listdir(expected_output_dir)
            
            # Check for HDF5 output file (new format)
            h5_file = os.path.join(expected_output_dir, 'ed_results.h5')
            if os.path.exists(h5_file):
                logging.warning(f"ED for cluster {cluster_id} crashed with exit code {e.returncode} but HDF5 output file exists - treating as success")
                return True
            
            if ed_options["thermo"]:
                # For thermodynamic calculations, check for thermo directory (legacy)
                thermo_dir = os.path.join(expected_output_dir, 'thermo')
                if os.path.exists(thermo_dir) and os.listdir(thermo_dir):
                    logging.warning(f"ED for cluster {cluster_id} crashed with exit code {e.returncode} but thermodynamic output files exist - treating as success")
                    return True
            
            # For other methods, check for any meaningful output files
            if output_files and any(f.endswith(('.dat', '.txt', '.h5')) or os.path.isdir(os.path.join(expected_output_dir, f)) for f in output_files):
                logging.warning(f"ED for cluster {cluster_id} crashed with exit code {e.returncode} but output files exist - treating as success")
                return True
        
        # Log the error with more specific information about the signal
        if e.returncode == -11:  # SIGSEGV
            logging.error(f"ED for cluster {cluster_id} crashed with SIGSEGV (segmentation fault) - computation may have completed but program crashed during cleanup")
        else:
            logging.error(f"Error running ED for cluster {cluster_id}: {e}")
        
        logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return False


def run_lb_ed_for_cluster(args):
    """Run ED for a single cluster in Lanczos-Boosted NLCE mode.
    
    For Lanczos-Boosted NLCE (Bhattaram & Khatami):
    - Small clusters (sites <= lb_site_threshold): Full ED with all eigenvalues
    - Large clusters (sites > lb_site_threshold): Partial Lanczos with N_low eigenvalues
    
    This is deterministic (no stochastic noise like FTLM) and works well for 
    low-to-intermediate temperatures where only low-energy states contribute
    to thermodynamics.
    
    Key difference from run_ed_for_cluster: We request --compute_eigenvectors=true
    so we can compute observables <n|A|n> for each eigenstate.
    """
    (cluster_id, order, ed_executable, ham_dir, ed_dir, lb_options) = args
    
    # Create output directory for ED results
    cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
    os.makedirs(cluster_ed_dir, exist_ok=True)
    
    # Set up ED command
    ham_subdir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
    
    if not os.path.exists(ham_subdir):
        logging.warning(f"[LB-NLCE] Hamiltonian directory not found for cluster {cluster_id}")
        return False
    
    # Get number of sites from the input file
    site_info_file = os.path.join(ham_subdir, f"*_site_info.dat")
    site_info_files = glob.glob(site_info_file)
    
    if not site_info_files:
        logging.warning(f"[LB-NLCE] Site info file not found for cluster {cluster_id}")
        return False
    
    # Count lines in site info file to get number of sites (excluding header lines)
    num_sites = 0
    with open(site_info_files[0], 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                num_sites += 1
    
    # Calculate Hilbert space dimension
    hilbert_dim = 2 ** num_sites
    
    # Decide method based on cluster size
    lb_site_threshold = lb_options.get("lb_site_threshold", 12)
    lb_n_eigenvalues = lb_options.get("lb_n_eigenvalues", 200)
    
    if num_sites <= lb_site_threshold:
        # Small cluster: Full ED
        n_eigs = "FULL"
        method = "FULL"
        logging.info(f"[LB-NLCE] Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): "
                    f"Full ED (all eigenvalues)")
    else:
        # Large cluster: Partial Lanczos
        # Ensure we don't request more eigenvalues than Hilbert space dimension
        n_eigs = min(lb_n_eigenvalues, hilbert_dim)
        method = "LANCZOS"
        logging.info(f"[LB-NLCE] Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): "
                    f"Partial Lanczos (N_low={n_eigs} eigenvalues)")
    
    # Build ED command - request eigenvalues with eigenvectors for observable computation
    cmd = [
        ed_executable,
        ham_subdir,
        f'--method={method}',
        f'--eigenvalues={n_eigs}',
        f'--output={cluster_ed_dir}/output',
        f'--num_sites={num_sites}',
        '--spin_length=0.5',
        '--symm',  # Auto-select best symmetry mode
        '--compute_eigenvectors',  # Needed for <n|O|n> in LB-NLCE
    ]
    
    # Add thermodynamic parameters - we compute them in post-processing for LB-NLCE
    # but still pass temp range for metadata
    if lb_options.get("thermo", False):
        cmd.extend([
            '--thermo',
            f'--temp_min={lb_options["temp_min"]}',
            f'--temp_max={lb_options["temp_max"]}',
            f'--temp_bins={lb_options["temp_bins"]}'
        ])
    
    if lb_options.get("measure_spin", False):
        cmd.append('--measure_spin')
    
    # Set ED_PYTHON environment variable
    env = os.environ.copy()
    env['ED_PYTHON'] = sys.executable
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        # Check if computation succeeded despite exit error
        expected_output_dir = os.path.join(cluster_ed_dir, 'output')
        
        if os.path.exists(expected_output_dir):
            h5_file = os.path.join(expected_output_dir, 'ed_results.h5')
            if os.path.exists(h5_file):
                logging.warning(f"[LB-NLCE] ED for cluster {cluster_id} crashed but HDF5 output exists - treating as success")
                return True
        
        if e.returncode == -11:
            logging.error(f"[LB-NLCE] ED for cluster {cluster_id} crashed with SIGSEGV")
        else:
            logging.error(f"[LB-NLCE] Error running ED for cluster {cluster_id}: {e}")
        
        logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automate NLCE workflow for pyrochlore lattice')
    
    # Parameters for the entire workflow
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--base_dir', type=str, default='./nlce_results', help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='../../../build/ED', help='Path to the ED executable')
    
    # Model parameters
    parser.add_argument('--Jxx', type=float, default=1.0, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, default=1.0, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz coupling')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], help='Field direction (x,y,z)')
    
    # ED parameters
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method (FULL, LANCZOS, etc.)')
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.001, help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    
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
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')

    parser.add_argument('--symmetrized', action='store_true', 
                       help='Legacy flag: force --symmetrized instead of --symm')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')

    # Random transverse field
    parser.add_argument('--random_field_width', type=float, default=0, help='Width of the random transverse field')

    # Automatic method and symmetry selection (default behavior)
    # Uses --symm flag which auto-selects between symmetrized and streaming-symmetry modes
    parser.add_argument('--no_auto_method', action='store_true',
                       help='Disable automatic method selection. By default, uses FULL for small '
                            'clusters and BLOCK_LANCZOS for larger ones.')
    parser.add_argument('--full_ed_threshold', type=int, default=12,
                       help='Site threshold for FULL vs BLOCK_LANCZOS (default: 12). '
                            'Clusters with more sites use BLOCK_LANCZOS.')
    parser.add_argument('--block_size', type=int, default=8,
                       help='Block size for BLOCK_LANCZOS (default: 8). '
                            'Should be >= expected degeneracy (e.g., 4-8 for Heisenberg).')
    
    # ========== Lanczos-Boosted NLCE Parameters ==========
    # Based on Bhattaram & Khatami method where large clusters use partial Lanczos
    # with only low-lying eigenvalues, rather than full ED or stochastic FTLM.
    parser.add_argument('--lanczos_boost', action='store_true',
                       help='Enable Lanczos-boosted NLCE mode. Large clusters use partial '
                            'Lanczos diagonalization (low-energy eigenstates only) instead '
                            'of full ED. Deterministic and noise-free, ideal for low-to-'
                            'intermediate temperatures.')
    parser.add_argument('--lb_site_threshold', type=int, default=12,
                       help='Site threshold for LB-NLCE (default: 12). Clusters with more '
                            'sites use partial Lanczos. Clusters with <= sites use full ED.')
    parser.add_argument('--lb_n_eigenvalues', type=int, default=200,
                       help='Number of low-lying eigenvalues to compute for large clusters '
                            'in LB-NLCE mode (default: 200). Should satisfy E_N - E_0 > 10*T_min '
                            'for temperature accuracy.')
    parser.add_argument('--lb_energy_window', type=float, default=None,
                       help='Alternative to --lb_n_eigenvalues: specify an energy window above '
                            'ground state. All eigenvalues with E - E_0 <= window are included. '
                            'Suggested: 10 * T_max for good accuracy.')
    parser.add_argument('--lb_check_convergence', action='store_true',
                       help='For LB-NLCE, check convergence by comparing results with '
                            'increasing numbers of eigenvalues.')

    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'nlce_workflow.log')
    setup_logging(log_file)
    
    # Define directory structure
    cluster_dir = os.path.join(args.base_dir, f'clusters_order_{args.max_order}')
    ham_dir = os.path.join(args.base_dir, f'hamiltonians_order_{args.max_order}')
    ed_dir = os.path.join(args.base_dir, f'ed_results_order_{args.max_order}')
    nlc_dir = os.path.join(args.base_dir, f'nlc_results_order_{args.max_order}')
    
    # Create directories if they don't exist
    for directory in [cluster_dir, ham_dir, ed_dir, nlc_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Generate clusters
    if not args.skip_cluster_gen:
        logging.info("="*80)
        logging.info("Step 1: Generating topologically distinct clusters with multiplicities")
        logging.info("="*80)
        
        cmd = [
            sys.executable,  # Use the same Python interpreter as the current script
            os.path.join(os.path.dirname(__file__), '..', 'prep', 'generate_pyrochlore_clusters.py'),
            f'--max_order={args.max_order}',
            f'--output_dir={cluster_dir}',
            '--visualize'  # Generate dual representation visualizations
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
            
            # Create output directory for this cluster
            cluster_ham_dir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
            os.makedirs(cluster_ham_dir, exist_ok=True)
            
            # Run helper_cluster.py (now in python/edlib/)
            cmd = [
                sys.executable,  # Use the same Python interpreter as the current script
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
                logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
                logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
    else:
        logging.info("Skipping Hamiltonian preparation step.")
    
    # Step 3: Run Exact Diagonalization for each cluster
    if not args.skip_ed:
        logging.info("="*80)
        logging.info("Step 3: Running Exact Diagonalization for each cluster")
        logging.info("="*80)
        
        if args.lanczos_boost:
            # ========== Lanczos-Boosted NLCE Mode ==========
            # Use partial Lanczos for large clusters (deterministic, no FTLM noise)
            logging.info("Using Lanczos-Boosted NLCE mode (Bhattaram & Khatami)")
            logging.info(f"  - Small clusters (<= {args.lb_site_threshold} sites): Full ED")
            logging.info(f"  - Large clusters (> {args.lb_site_threshold} sites): Partial Lanczos ({args.lb_n_eigenvalues} eigenvalues)")
            
            lb_options = {
                "lb_site_threshold": args.lb_site_threshold,
                "lb_n_eigenvalues": args.lb_n_eigenvalues,
                "lb_energy_window": args.lb_energy_window,
                "thermo": args.thermo,
                "temp_min": args.temp_min,
                "temp_max": args.temp_max,
                "temp_bins": args.temp_bins,
                "measure_spin": args.measure_spin,
            }
            
            # Prepare tasks for LB-NLCE
            lb_ed_tasks = []
            for cluster_id, order, _ in clusters:
                lb_ed_tasks.append((cluster_id, order, args.ed_executable, ham_dir, ed_dir, lb_options))
            
            if args.parallel:
                logging.info(f"Running LB-NLCE ED in parallel with {args.num_cores} cores")
                with multiprocessing.Pool(processes=args.num_cores) as pool:
                    results = list(tqdm(
                        pool.imap(run_lb_ed_for_cluster, lb_ed_tasks),
                        total=len(lb_ed_tasks),
                        desc="Running LB-NLCE ED"
                    ))
                success_count = sum(results)
                logging.info(f"LB-NLCE ED completed for {success_count} of {len(lb_ed_tasks)} clusters")
            else:
                for task in tqdm(lb_ed_tasks, desc="Running LB-NLCE ED"):
                    run_lb_ed_for_cluster(task)
        
        else:
            # ========== Standard NLCE Mode ==========
            # Prepare ED options with automatic method and symmetry selection
            ed_options = {
                "method": args.method,
                "thermo": args.thermo,
                "temp_min": args.temp_min,
                "temp_max": args.temp_max,
                "temp_bins": args.temp_bins,
                "measure_spin": args.measure_spin,
                # Automatic method selection (FULL vs BLOCK_LANCZOS)
                "auto_method": not args.no_auto_method,
                "full_ed_threshold": args.full_ed_threshold,
                "block_size": args.block_size,
            }
            
            logging.info(f"NLCE ED Configuration:")
            logging.info(f"  - Method selection: {'automatic' if ed_options['auto_method'] else 'manual (' + args.method + ')'}")
            if ed_options['auto_method']:
                logging.info(f"  - FULL for clusters with <= {args.full_ed_threshold} sites")
                logging.info(f"  - BLOCK_LANCZOS (block_size={args.block_size}) for larger clusters")
            logging.info(f"  - Symmetry: --symm (auto-select best mode)")
            
            # Prepare arguments for each cluster
            ed_tasks = []
            for cluster_id, order, _ in clusters:
                ed_tasks.append((cluster_id, order, args.ed_executable, ham_dir, ed_dir, ed_options, args.symmetrized))
            
            if args.parallel:
                logging.info(f"Running ED in parallel with {args.num_cores} cores")
                with multiprocessing.Pool(processes=args.num_cores) as pool:
                    results = list(tqdm(
                        pool.imap(run_ed_for_cluster, ed_tasks),
                        total=len(ed_tasks),
                        desc="Running ED"
                    ))
                
                # Check results
                success_count = sum(results)
                logging.info(f"ED completed for {success_count} of {len(ed_tasks)} clusters")
            else:
                # Run sequentially
                for task in tqdm(ed_tasks, desc="Running ED"):
                    run_ed_for_cluster(task)
    else:
        logging.info("Skipping Exact Diagonalization step.")
    
    # Step 3.5: Plot thermodynamic data for each cluster
    if args.thermo and not args.skip_ed and not args.method == "mTPQ":
        logging.info("="*80)
        logging.info("Step 3.5: Plotting thermodynamic data for each cluster")
        logging.info("="*80)
        
        # Create directory for thermodynamic plots
        thermo_plots_dir = os.path.join(args.base_dir, f'thermo_plots_order_{args.max_order}')
        os.makedirs(thermo_plots_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            
            # Iterate through all clusters
            for cluster_id, order, _ in tqdm(clusters, desc="Plotting thermodynamic data"):
                cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
                output_dir = os.path.join(cluster_ed_dir, "output")
                
                # Try HDF5 file first (new format)
                h5_file = os.path.join(output_dir, "ed_results.h5")
                thermo_data = None
                
                if HAS_H5PY and os.path.exists(h5_file):
                    try:
                        with h5py.File(h5_file, 'r') as f:
                            # Check for thermodynamics group
                            if '/thermodynamics' in f:
                                thermo_grp = f['/thermodynamics']
                                if 'temperatures' in thermo_grp:
                                    thermo_data = {
                                        'T': thermo_grp['temperatures'][:],
                                        'energy': thermo_grp['energy'][:] if 'energy' in thermo_grp else None,
                                        'specific_heat': thermo_grp['specific_heat'][:] if 'specific_heat' in thermo_grp else None,
                                        'entropy': thermo_grp['entropy'][:] if 'entropy' in thermo_grp else None,
                                        'free_energy': thermo_grp['free_energy'][:] if 'free_energy' in thermo_grp else None
                                    }
                            # Also try FTLM averaged
                            elif '/ftlm/averaged' in f:
                                ftlm_grp = f['/ftlm/averaged']
                                if 'temperatures' in ftlm_grp:
                                    thermo_data = {
                                        'T': ftlm_grp['temperatures'][:],
                                        'energy': ftlm_grp['energy'][:] if 'energy' in ftlm_grp else None,
                                        'specific_heat': ftlm_grp['specific_heat'][:] if 'specific_heat' in ftlm_grp else None,
                                        'entropy': ftlm_grp['entropy'][:] if 'entropy' in ftlm_grp else None,
                                        'free_energy': ftlm_grp['free_energy'][:] if 'free_energy' in ftlm_grp else None
                                    }
                    except Exception as e:
                        logging.warning(f"Error reading HDF5 for cluster {cluster_id}: {e}")
                
                # Fall back to legacy text file
                if thermo_data is None:
                    thermo_file = os.path.join(output_dir, "thermo/thermo_data.txt")
                    if not os.path.exists(thermo_file):
                        logging.warning(f"No thermodynamic data found for cluster {cluster_id}")
                        continue
                    
                    try:
                        data = np.loadtxt(thermo_file, comments='#')
                        data = np.atleast_2d(data)
                        thermo_data = {
                            'T': data[:, 0],
                            'energy': data[:, 1] if data.shape[1] > 1 else None,
                            'specific_heat': data[:, 2] if data.shape[1] > 2 else None,
                            'entropy': data[:, 3] if data.shape[1] > 3 else None,
                            'free_energy': data[:, 4] if data.shape[1] > 4 else None
                        }
                    except Exception as e:
                        logging.error(f"Error reading text file for cluster {cluster_id}: {e}")
                        continue
                
                if thermo_data is None:
                    logging.warning(f"No thermodynamic data found for cluster {cluster_id}")
                    continue
                
                try:
                    T = thermo_data['T']
                    sort_idx = np.argsort(T)
                    T = T[sort_idx]

                    # Create plots
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f"Thermodynamic Properties for Cluster {cluster_id} (Order {order})")

                    # Plot energy
                    if thermo_data['energy'] is not None:
                        axs[0, 0].plot(T, thermo_data['energy'][sort_idx], 'r-')
                    axs[0, 0].set_xlabel("Temperature")
                    axs[0, 0].set_ylabel("Energy")
                    axs[0, 0].set_xscale('log')
                    axs[0, 0].grid(True)

                    # Plot specific heat
                    if thermo_data['specific_heat'] is not None:
                        axs[0, 1].plot(T, thermo_data['specific_heat'][sort_idx], 'b-')
                    axs[0, 1].set_xlabel("Temperature")
                    axs[0, 1].set_ylabel("Specific Heat")
                    axs[0, 1].set_xscale('log')
                    axs[0, 1].grid(True)

                    # Plot entropy
                    if thermo_data['entropy'] is not None:
                        axs[1, 0].plot(T, thermo_data['entropy'][sort_idx], 'g-')
                    axs[1, 0].set_xlabel("Temperature")
                    axs[1, 0].set_ylabel("Entropy")
                    axs[1, 0].set_xscale('log')
                    axs[1, 0].grid(True)

                    # Plot free energy
                    if thermo_data['free_energy'] is not None:
                        axs[1, 1].plot(T, thermo_data['free_energy'][sort_idx], 'm-')
                    axs[1, 1].set_xlabel("Temperature")
                    axs[1, 1].set_ylabel("Free Energy")
                    axs[1, 1].set_xscale('log')
                    axs[1, 1].grid(True)

                    # Save plot
                    plt.tight_layout()
                    plt.savefig(os.path.join(thermo_plots_dir, f"thermo_cluster_{cluster_id}_order_{order}.png"))
                    plt.close(fig)

                    logging.info(f"Thermodynamic plots created for cluster {cluster_id}")
                    
                except Exception as e:
                    logging.error(f"Error plotting thermodynamic data for cluster {cluster_id}: {e}")
                    logging.error(traceback.format_exc())
        
        except ImportError:
            logging.error("Matplotlib not installed. Skipping thermodynamic plots.")
    elif args.method == 'mTPQ':
        
        logging.info("="*80)
        logging.info("Step 3.5: Plotting mTPQ thermodynamic data for each cluster")
        logging.info("="*80)

        # Create directory for thermodynamic plots
        thermo_plots_dir = os.path.join(args.base_dir, f'thermo_plots_order_{args.max_order}')
        os.makedirs(thermo_plots_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            
            # Iterate through all clusters
            for cluster_id, order, _ in tqdm(clusters, desc="Plotting mTPQ thermodynamic data"):
                cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
                output_dir = os.path.join(cluster_ed_dir, "output")
                
                # Check if directory exists
                if not os.path.exists(output_dir):
                    logging.warning(f"No output directory found for cluster {cluster_id}")
                    continue
                
                all_temps = None
                all_energies = None
                all_variances = None
                
                # Try HDF5 file first (new format)
                h5_file = os.path.join(output_dir, "ed_results.h5")
                if HAS_H5PY and os.path.exists(h5_file):
                    try:
                        with h5py.File(h5_file, 'r') as f:
                            if '/tpq/averaged' in f and 'thermodynamics' in f['/tpq/averaged']:
                                tpq_data = f['/tpq/averaged/thermodynamics'][:]
                                # Format: beta, energy, variance, doublon, step
                                betas = tpq_data[:, 0]
                                all_temps = 1.0 / betas
                                all_energies = tpq_data[:, 1]
                                all_variances = tpq_data[:, 2] * betas**2
                            elif '/tpq/samples/sample_0/thermodynamics' in f:
                                # Read from first sample
                                tpq_data = f['/tpq/samples/sample_0/thermodynamics'][:]
                                betas = tpq_data[:, 0]
                                all_temps = 1.0 / betas
                                all_energies = tpq_data[:, 1]
                                all_variances = tpq_data[:, 2] * betas**2
                    except Exception as e:
                        logging.warning(f"Error reading HDF5 TPQ data for cluster {cluster_id}: {e}")
                
                # Fall back to legacy text file format
                if all_temps is None:
                    ss_file = os.path.join(output_dir, "SS_rand0.dat")
                    if not os.path.exists(ss_file):
                        logging.warning(f"No TPQ data found for cluster {cluster_id}")
                        continue
                        
                    SS_data = np.loadtxt(ss_file, unpack=True, skiprows=2)
                    all_temps = 1.0 / SS_data[0]
                    all_energies = SS_data[1] 
                    all_variances = SS_data[2] * SS_data[0]**2

                logging.info(f"Loaded TPQ data for cluster {cluster_id}")   

                fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                fig.suptitle(f"mTPQ Thermodynamic Properties for Cluster {cluster_id} (Order {order})")

                # Plot energy
                axs[0].plot(all_temps, all_energies, label='Energy')
                axs[0].set_xlabel("Temperature")
                axs[0].set_ylabel("Energy per site")
                axs[0].set_xscale('log')
                axs[0].grid(True)
                axs[0].legend()
                
                # Plot specific heat
                axs[1].plot(all_temps, all_variances, label='Specific Heat')
                axs[1].set_xlabel("Temperature")
                axs[1].set_ylabel("Specific Heat")
                axs[1].set_xscale('log')
                axs[1].grid(True)
                axs[1].legend()
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(thermo_plots_dir, f"mTPQ_thermo_cluster_{cluster_id}_order_{order}.png"))
                plt.close(fig)

                logging.info(f"mTPQ thermodynamic plots and data created for cluster {cluster_id}")

        except ImportError:
            logging.error("Matplotlib not installed. Skipping mTPQ thermodynamic plots.")
        except Exception as e:
            logging.error(f"Error in mTPQ thermodynamic analysis: {e}")
            logging.error(traceback.format_exc())

    # Step 4: Perform NLCE summation
    if not args.skip_nlc:
        logging.info("="*80)
        logging.info("Step 4: Performing NLCE summation")
        logging.info("="*80)
        
        if args.lanczos_boost:
            # ========== Lanczos-Boosted NLCE Summation ==========
            logging.info("Using Lanczos-Boosted NLCE summation (truncated thermodynamics)")
            nlc_params = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), 'NLC_sum_LB.py'),
                f'--cluster_dir={cluster_info_dir}',
                f'--eigenvalue_dir={ed_dir}',
                f'--output_dir={nlc_dir}',
                '--plot',
                f'--temp_min={args.temp_min}',
                f'--temp_max={args.temp_max}',
                f'--temp_bins={args.temp_bins}',
                f'--resummation_method=auto',
                f'--lb_energy_tolerance=10.0',
            ]
        elif args.method == 'mTPQ':
            logging.info("Using mTPQ method for NLCE summation")
            # Add mTPQ specific parameters here if needed
            nlc_params = [
                sys.executable,  # Use the same Python interpreter
                os.path.join(os.path.dirname(__file__), 'NLC_sum_TPQ.py'),
                f'--cluster_dir={cluster_info_dir}',
                f'--eigenvalue_dir={ed_dir}',
                f'--output_dir={nlc_dir}',
                '--plot',
                f'--temp_min={args.temp_min}',
                f'--temp_max={args.temp_max}',
                f'--temp_bins={args.temp_bins}'
            ]
        else:
            nlc_params = [
                sys.executable,  # Use the same Python interpreter
                os.path.join(os.path.dirname(__file__), 'NLC_sum.py'),
                f'--cluster_dir={cluster_info_dir}',
                f'--eigenvalue_dir={ed_dir}',
                f'--output_dir={nlc_dir}',
                '--plot',
                f'--temp_min={args.temp_min}',
                f'--temp_max={args.temp_max}',
                f'--temp_bins={args.temp_bins}',
                f'--resummation_method=auto'
            ]
            
        if args.SI_units:
            nlc_params.append('--SI_units')
            
        if args.order_cutoff:
            nlc_params.append(f'--order_cutoff={args.order_cutoff}')
        
        if args.measure_spin:
            nlc_params.append('--measure_spin')
        
        logging.info(f"Running command: {' '.join(nlc_params)}")
        try:
            subprocess.run(nlc_params, check=True)
            logging.info("NLCE summation completed successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error in NLCE summation: {e}")
    else:
        logging.info("Skipping NLCE summation step.")
    
    logging.info("="*80)
    logging.info("NLCE workflow completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")