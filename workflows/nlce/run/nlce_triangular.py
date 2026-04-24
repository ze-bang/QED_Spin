#!/usr/bin/env python3
"""NLCE workflow driver: triangular lattice, full / ScaLAPACK ED.

Sister to `nlce.py` for the triangular lattice. Supports two NLCE
expansion schemes:

* Triangle-based (the default): order = number of triangles. Gives far
  fewer clusters per order; useful for frustrated systems.
* Site-based (`--site_based`): order = number of sites.

Models supported via `--model`:
  - `xxz_j1j2`   (default): J1-J2 XXZ
  - `kitaev`            : J-K-Γ-Γ' Kitaev
  - `anisotropic`       : YbMgGaO4-style anisotropic exchange
"""

import argparse
import logging
import multiprocessing
import os
import subprocess
import sys

from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from workflows.nlce._common import (  # noqa: E402
    DEFAULT_ED_PATH,
    EDOptions,
    build_ed_command,
    count_sites_in_info_file,
    get_cluster_files,
    run_ed_subprocess,
    setup_logging,
)


def run_ed_for_cluster(args):
    """Run ED for a single triangular cluster.

    Two key triangular-specific knobs vs `nlce.py`:

    * `symm_threshold`: only attach `--symm` for clusters with strictly
      more sites than this threshold (default 13). Smaller clusters
      tend not to benefit from the symmetry-projected basis.
    * `streaming_symmetry`: opt-in to the streaming-symmetry kernel
      with `<ham_subdir>/basis_cache` as the pre-baked orbit basis.
    * For `num_sites <= 8` we pin `OMP_NUM_THREADS=1` to dodge an
      OpenMP race condition on very small matrices.
    """
    cluster_id, order, ed_executable, ham_dir, ed_dir, ed_options, use_gpu = args

    cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
    os.makedirs(os.path.join(cluster_ed_dir, 'output'), exist_ok=True)

    ham_subdir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
    if not os.path.exists(ham_subdir):
        logging.warning(f"Hamiltonian directory not found for cluster {cluster_id}")
        return False

    num_sites = count_sites_in_info_file(ham_subdir)
    if num_sites is None:
        logging.warning(f"Site info file not found for cluster {cluster_id}")
        return False

    use_symm = (num_sites > ed_options.get("symm_threshold", 13))
    streaming_symmetry = ed_options.get("streaming_symmetry", False)
    basis_cache = os.path.join(ham_subdir, 'basis_cache') if streaming_symmetry else None

    options = EDOptions(
        method=ed_options.get("method", "FULL"),
        eigenvalues="FULL",
        thermo=ed_options.get("thermo", False),
        temp_min=ed_options.get("temp_min", 0.1),
        temp_max=ed_options.get("temp_max", 10.0),
        temp_bins=ed_options.get("temp_bins", 100),
        measure_spin=ed_options.get("measure_spin", False),
        use_symm=use_symm,
        streaming_symmetry=streaming_symmetry,
        basis_cache_dir=basis_cache,
    )

    cmd = build_ed_command(
        ed_executable=ed_executable,
        ham_subdir=ham_subdir,
        output_dir=cluster_ed_dir,
        num_sites=num_sites,
        options=options,
        scalapack_threshold=ed_options.get("scalapack_threshold", 16),
        use_scalapack=ed_options.get("use_scalapack", True),
    )

    method = next((flag.split('=', 1)[1] for flag in cmd if flag.startswith('--method=')), 'FULL')
    logging.info(
        "Cluster %d (%d sites, dim=2^%d): %s%s",
        cluster_id, num_sites, num_sites, method,
        " with --symm" if use_symm else "",
    )

    extra_env = {}
    if num_sites <= 8:
        extra_env["OMP_NUM_THREADS"] = "1"

    return run_ed_subprocess(
        cmd,
        output_root=cluster_ed_dir,
        extra_env=extra_env,
        log_tag=f"ED:cluster_{cluster_id}",
    )


def main():
    parser = argparse.ArgumentParser(description='Automate NLCE workflow for triangular lattice')
    
    # Parameters for the entire workflow
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--base_dir', type=str, default='./nlce_triangular_results', help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default=DEFAULT_ED_PATH,
                        help='Path to the ED executable (defaults to <repo_root>/build/ED).')
    
    # Model parameters for triangular lattice
    parser.add_argument('--J1', type=float, default=1.0, help='Nearest-neighbor exchange coupling')
    parser.add_argument('--J2', type=float, default=0.0, help='Next-nearest-neighbor exchange coupling')
    parser.add_argument('--Jz_ratio', type=float, default=1.0, help='Jz/Jxy ratio for XXZ model')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field strength')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1], 
                       help='Field direction (x,y,z), default is out-of-plane')
    parser.add_argument('--model', type=str, default='xxz_j1j2', 
                       choices=['xxz_j1j2', 'kitaev', 'anisotropic'],
                       help='Spin model type')
    
    # Anisotropic exchange model parameters (YbMgGaO4-type)
    parser.add_argument('--Jzz', type=float, default=None, help='J_zz for anisotropic model')
    parser.add_argument('--Jpm', type=float, default=None, help='J_± for anisotropic model')
    parser.add_argument('--Jpmpm', type=float, default=None, help='J_±± for anisotropic model')
    parser.add_argument('--Jzpm', type=float, default=None, help='J_z± for anisotropic model')
    
    # JKΓΓ' (Kitaev) model parameters
    parser.add_argument('--Gamma', type=float, default=None, help='Γ off-diagonal symmetric exchange for kitaev model')
    parser.add_argument('--Gamma_prime', type=float, default=None, help="Γ' off-diagonal exchange for kitaev model")
    
    # Anisotropic g-tensor for Zeeman term
    parser.add_argument('--g_ab', type=float, default=2.0,
                       help='In-plane g-factor for Zeeman coupling (default: 2.0)')
    parser.add_argument('--g_c', type=float, default=2.0,
                       help='Out-of-plane (c-axis) g-factor for Zeeman coupling (default: 2.0)')
    
    # ED parameters
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method')
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.1, help='Minimum temperature (default 0.1 - NLCE poorly converges at lower T for frustrated systems)')
    parser.add_argument('--temp_max', type=float, default=10.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    parser.add_argument('--temp_points_file', type=str, default=None,
                       help='File containing explicit temperature points (one per line, in Kelvin). '
                            'Overrides --temp_min/--temp_max/--temp_bins for NLCE summation.')
    parser.add_argument('--resummation', type=str, default='euler',
                       choices=['none', 'euler', 'wynn', 'wynn_multi', 'entropy_derived'],
                       help='Resummation method: none, euler, wynn, wynn_multi (multi-start median), '
                            'or entropy_derived (C = T dS/dT from resummed entropy, best for frustrated systems)')
    
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
                       help='Convert to SI units: specific heat in J/(mol·K).')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')

    # ScaLAPACK distributed diagonalization for large clusters
    parser.add_argument('--scalapack_threshold', type=int, default=16,
                       help='Site threshold for switching to ScaLAPACK (default: 16). '
                            'Clusters with >= sites use SCALAPACK_MIXED for distributed diagonalization.')
    parser.add_argument('--no_scalapack', action='store_true',
                       help='Disable ScaLAPACK - always use standard FULL diagonalization.')
    parser.add_argument('--symm_threshold', type=int, default=13,
                       help='Site threshold for using --symm flag (default: 13)')
    
    # Streaming-symmetry diagonalization with basis caching
    parser.add_argument('--streaming-symmetry', action='store_true',
                       help='Use streaming-symmetry diagonalization (exploits spatial automorphisms). '
                            'Automatically precomputes and caches the orbit basis for all clusters '
                            'before running ED, so the basis is reused across fitting iterations.')
    parser.add_argument('--skip_basis_precompute', action='store_true',
                       help='Skip orbit basis precomputation (assumes basis cache already exists). '
                            'Only meaningful with --streaming-symmetry.')

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
    
    # Create directories (skip if already exists as symlink or dir)
    for directory in [cluster_dir, ham_dir, ed_dir, nlc_dir]:
        if os.path.islink(directory) or os.path.isdir(directory):
            continue
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
        elif args.model == 'kitaev':
            logging.info(f"Model: {args.model}, J={args.J1}, K={args.J2}, Γ={args.Gamma}, Γ'={args.Gamma_prime}, h={args.h}")
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
            
            # Add JKΓΓ' (Kitaev) model parameters if specified
            if args.Gamma is not None:
                cmd.extend(['--Gamma', str(args.Gamma)])
            if args.Gamma_prime is not None:
                cmd.extend(['--Gamma_prime', str(args.Gamma_prime)])
            
            # g-tensor parameters
            cmd.extend(['--g_ab', str(args.g_ab)])
            cmd.extend(['--g_c', str(args.g_c)])
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error preparing Hamiltonian for cluster {cluster_id}: {e}")
                logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
                logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
    else:
        logging.info("Skipping Hamiltonian preparation step.")
    
    # Step 2.5: Precompute orbit basis for all clusters (if --streaming-symmetry)
    # The orbit basis depends on the cluster geometry AND on which operator types
    # appear on each bond (encoded as edge labels by the automorphism finder).
    # If a coupling is exactly zero, that bond type vanishes and the
    # automorphism group may enlarge — producing a basis incompatible with
    # nonzero values of that coupling.  When using the fitter, a dedicated
    # basis-seeding pass with all-nonzero couplings should be run BEFORE the
    # optimizer loop to ensure the cached basis is valid for all parameter
    # combinations (see nlc_fit_triangular.py).
    if args.streaming_symmetry and not args.skip_basis_precompute:
        logging.info("="*80)
        logging.info("Step 2.5: Precomputing orbit basis for streaming-symmetry diagonalization")
        logging.info("="*80)
        
        def _precompute_basis_for_cluster(task_args):
            """Precompute orbit basis for a single cluster."""
            cluster_id, order, ed_executable, ham_dir_local = task_args
            ham_subdir = os.path.join(ham_dir_local, f'cluster_{cluster_id}_order_{order}')
            if not os.path.exists(ham_subdir):
                logging.warning(f"Hamiltonian dir not found for cluster {cluster_id}")
                return False
            
            # Check if basis cache already exists
            basis_cache = os.path.join(ham_subdir, 'basis_cache')
            if os.path.isdir(basis_cache) and glob.glob(os.path.join(basis_cache, '*.h5')):
                logging.debug(f"Basis cache already exists for cluster {cluster_id} — skipping")
                return True
            
            # Count sites
            site_info_files = glob.glob(os.path.join(ham_subdir, '*_site_info.dat'))
            if not site_info_files:
                logging.warning(f"Site info file not found for cluster {cluster_id}")
                return False
            num_sites = 0
            with open(site_info_files[0], 'r') as f:
                for line in f:
                    if not line.startswith('#') and line.strip():
                        num_sites += 1
            
            cmd = [
                ed_executable,
                ham_subdir,
                '--precompute-basis',
                f'--num_sites={num_sites}',
                '--spin_length=0.5',
            ]
            
            env = os.environ.copy()
            env['ED_PYTHON'] = sys.executable
            if num_sites <= 8:
                env['OMP_NUM_THREADS'] = '1'
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, env=env)
                return True
            except subprocess.CalledProcessError as e:
                logging.error(f"Basis precomputation failed for cluster {cluster_id}: {e}")
                logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
                return False
        
        precompute_tasks = [
            (cid, order, args.ed_executable, ham_dir)
            for cid, order, _ in clusters
        ]
        
        if args.parallel:
            logging.info(f"Precomputing basis in parallel with {args.num_cores} cores")
            with multiprocessing.Pool(processes=args.num_cores) as pool:
                results = list(tqdm(
                    pool.imap(_precompute_basis_for_cluster, precompute_tasks),
                    total=len(precompute_tasks),
                    desc="Precomputing basis"
                ))
            success_count = sum(results)
            logging.info(f"Basis precomputed for {success_count}/{len(precompute_tasks)} clusters")
        else:
            for task in tqdm(precompute_tasks, desc="Precomputing basis"):
                _precompute_basis_for_cluster(task)
        
        logging.info("Basis precomputation complete — cached to each cluster's basis_cache/ directory")
    elif args.streaming_symmetry and args.skip_basis_precompute:
        logging.info("Skipping basis precomputation (--skip_basis_precompute). "
                     "Assuming basis cache already exists.")
    
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
            "streaming_symmetry": args.streaming_symmetry,
        }
        
        use_gpu = (args.method.upper() == 'FULL_GPU')  # GPU used only for FULL_GPU method
        
        logging.info(f"NLCE ED Configuration:")
        if args.method.upper() == 'FULL_GPU':
            logging.info(f"  - Method: FULL_GPU (GPU-accelerated dense diagonalization)")
        elif not args.no_scalapack:
            logging.info(f"  - Small clusters (< {args.scalapack_threshold} sites): FULL diagonalization")
            logging.info(f"  - Large clusters (>= {args.scalapack_threshold} sites): SCALAPACK_MIXED (distributed)")
        else:
            logging.info(f"  - Method: FULL diagonalization (ScaLAPACK disabled)")
        logging.info(f"  - Symmetry: --symm for clusters with > {args.symm_threshold} sites")
        if args.streaming_symmetry:
            logging.info(f"  - Streaming-symmetry diagonalization: ENABLED (orbit basis cached)")
        
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
            f'--max_order={order_cutoff}',
            f'--resummation={args.resummation}',
        ]
        
        # Temperature grid: explicit file takes priority over min/max/bins
        if args.temp_points_file:
            cmd.append(f'--temp_points_file={args.temp_points_file}')
        else:
            cmd.extend([
                f'--temp_min={args.temp_min}',
                f'--temp_max={args.temp_max}',
                f'--temp_bins={args.temp_bins}',
            ])
        
        if args.measure_spin:
            cmd.append('--measure_spin')
        
        if args.SI_units:
            cmd.append('--SI_units')
        
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
