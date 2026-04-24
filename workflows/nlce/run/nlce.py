#!/usr/bin/env python3
"""NLCE workflow driver: pyrochlore lattice, full / Lanczos-boosted ED.

Orchestrates the full Numerical Linked Cluster Expansion pipeline:
  1. Generate topologically distinct clusters on the pyrochlore lattice
     (via `prep/generate_pyrochlore_clusters.py`).
  2. Prepare per-cluster Hamiltonian parameter files
     (via `python/edlib/helper_cluster.py`).
  3. Diagonalize each cluster via the canonical `./ED` binary.
  4. Perform NLCE summation (via `NLC_sum.py` or `NLC_sum_LB.py`).

Shared infrastructure (logging, cluster discovery, ED-runner subprocess
plumbing, HDF5/text fallback readers) lives in
`workflows.nlce._common`. This file only owns the workflow
orchestration -- the *which clusters / which model / which
post-processing* decisions.
"""

import argparse
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm

# Make `workflows.nlce._common` importable when this script is run
# directly (the typical NLCE invocation pattern). The repo root is two
# levels up from `workflows/nlce/run/`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from workflows.nlce._common import (  # noqa: E402
    DEFAULT_ED_PATH,
    EDOptions,
    build_ed_command,
    count_sites_in_info_file,
    get_cluster_files,
    load_thermo_dataset,
    load_tpq_thermo_dataset,
    run_ed_subprocess,
    setup_logging,
)

def run_ed_for_cluster(args):
    """Run ED for a single cluster.

    Hands off to `_common.build_ed_command` for argv assembly (which
    handles the FULL -> SCALAPACK_MIXED auto-promotion for large
    clusters) and `_common.run_ed_subprocess` for the actual launch
    plus exit-code-vs-output reconciliation.
    """
    cluster_id, order, ed_executable, ham_dir, ed_dir, ed_options, symmetrized, use_gpu = args

    cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
    os.makedirs(cluster_ed_dir, exist_ok=True)

    ham_subdir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
    if not os.path.exists(ham_subdir):
        logging.warning(f"Hamiltonian directory not found for cluster {cluster_id}")
        return False

    num_sites = count_sites_in_info_file(ham_subdir)
    if num_sites is None:
        logging.warning(f"Site info file not found for cluster {cluster_id}")
        return False

    options = EDOptions(
        method=ed_options.get("method", "FULL"),
        eigenvalues="FULL",
        thermo=ed_options.get("thermo", False),
        temp_min=ed_options.get("temp_min", 0.001),
        temp_max=ed_options.get("temp_max", 20.0),
        temp_bins=ed_options.get("temp_bins", 100),
        measure_spin=ed_options.get("measure_spin", False),
        symmetrized=symmetrized,
        use_symm=not symmetrized,
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

    logging.info(
        "Cluster %d (%d sites, dim=2^%d): %s",
        cluster_id, num_sites, num_sites,
        next((flag.split('=', 1)[1] for flag in cmd if flag.startswith('--method=')), 'FULL'),
    )

    return run_ed_subprocess(
        cmd,
        output_root=cluster_ed_dir,
        log_tag=f"ED:cluster_{cluster_id}",
    )


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

    cluster_ed_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}')
    os.makedirs(cluster_ed_dir, exist_ok=True)

    ham_subdir = os.path.join(ham_dir, f'cluster_{cluster_id}_order_{order}')
    if not os.path.exists(ham_subdir):
        logging.warning(f"[LB-NLCE] Hamiltonian directory not found for cluster {cluster_id}")
        return False

    num_sites = count_sites_in_info_file(ham_subdir)
    if num_sites is None:
        logging.warning(f"[LB-NLCE] Site info file not found for cluster {cluster_id}")
        return False

    hilbert_dim = 2 ** num_sites
    lb_site_threshold = lb_options.get("lb_site_threshold", 12)
    lb_n_eigenvalues = lb_options.get("lb_n_eigenvalues", 200)

    if num_sites <= lb_site_threshold:
        method = "FULL"
        n_eigs: object = "FULL"
        logging.info(f"[LB-NLCE] Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): full ED")
    else:
        method = "LANCZOS"
        n_eigs = min(lb_n_eigenvalues, hilbert_dim)
        logging.info(
            f"[LB-NLCE] Cluster {cluster_id} ({num_sites} sites, dim={hilbert_dim}): "
            f"partial Lanczos (N_low={n_eigs})"
        )

    options = EDOptions(
        method=method,
        eigenvalues=str(n_eigs),
        thermo=lb_options.get("thermo", False),
        temp_min=lb_options.get("temp_min", 0.001),
        temp_max=lb_options.get("temp_max", 20.0),
        temp_bins=lb_options.get("temp_bins", 100),
        measure_spin=lb_options.get("measure_spin", False),
        use_symm=True,
        # LB-NLCE post-processing needs <n|O|n>, hence eigenvectors:
        extra_flags=["--compute_eigenvectors"],
    )

    cmd = build_ed_command(
        ed_executable=ed_executable,
        ham_subdir=ham_subdir,
        output_dir=cluster_ed_dir,
        num_sites=num_sites,
        options=options,
        # LB-NLCE never wants the FULL -> SCALAPACK_MIXED auto-promotion;
        # the method is already explicitly chosen above.
        scalapack_threshold=10**9,
        use_scalapack=False,
    )

    return run_ed_subprocess(
        cmd,
        output_root=cluster_ed_dir,
        log_tag=f"LB-NLCE:cluster_{cluster_id}",
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automate NLCE workflow for pyrochlore lattice')
    
    # Parameters for the entire workflow
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--base_dir', type=str, default='./nlce_results', help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default=DEFAULT_ED_PATH,
                       help='Path to the ED executable (defaults to <repo_root>/build/ED).')
    
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

    # ScaLAPACK distributed diagonalization for large clusters
    parser.add_argument('--scalapack_threshold', type=int, default=16,
                       help='Site threshold for switching to ScaLAPACK (default: 16). '
                            'Clusters with >= sites use SCALAPACK_MIXED for distributed diagonalization.')
    parser.add_argument('--no_scalapack', action='store_true',
                       help='Disable ScaLAPACK - always use standard FULL diagonalization.')
    
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
            # Prepare ED options with ScaLAPACK for large clusters
            ed_options = {
                "method": args.method,
                "thermo": args.thermo,
                "temp_min": args.temp_min,
                "temp_max": args.temp_max,
                "temp_bins": args.temp_bins,
                "measure_spin": args.measure_spin,
                "scalapack_threshold": args.scalapack_threshold,
                "use_scalapack": not args.no_scalapack,
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
            logging.info(f"  - Symmetry: --symm (auto-select best mode)")
            
            # Prepare arguments for each cluster
            ed_tasks = []
            for cluster_id, order, _ in clusters:
                ed_tasks.append((cluster_id, order, args.ed_executable, ham_dir, ed_dir, ed_options, args.symmetrized, use_gpu))
            
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
    
    # Step 3.5: Plot per-cluster thermodynamic data (full / FTLM / TPQ).
    # Reads via the unified `_common.load_*_thermo_dataset` helpers.
    is_tpq = args.method == 'mTPQ'
    if (args.thermo and not args.skip_ed) or is_tpq:
        logging.info("=" * 80)
        logging.info(
            "Step 3.5: Plotting per-cluster %s thermodynamic data",
            "TPQ" if is_tpq else "thermal",
        )
        logging.info("=" * 80)

        thermo_plots_dir = os.path.join(args.base_dir, f'thermo_plots_order_{args.max_order}')
        os.makedirs(thermo_plots_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.error("Matplotlib not installed. Skipping thermodynamic plots.")
            plt = None

        if plt is not None:
            for cluster_id, order, _ in tqdm(clusters, desc="Plotting thermo"):
                output_dir = os.path.join(ed_dir, f'cluster_{cluster_id}_order_{order}', 'output')
                data = load_tpq_thermo_dataset(output_dir) if is_tpq else load_thermo_dataset(output_dir)
                if data is None:
                    logging.warning(f"No thermodynamic data found for cluster {cluster_id}")
                    continue

                try:
                    T = np.asarray(data['T'])
                    sort_idx = np.argsort(T)
                    T = T[sort_idx]

                    if is_tpq:
                        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
                        fig.suptitle(f"mTPQ thermodynamics: cluster {cluster_id} (order {order})")
                        axs[0].plot(T, np.asarray(data['energy'])[sort_idx])
                        axs[0].set(xlabel="Temperature", ylabel="Energy per site", xscale='log')
                        axs[0].grid(True)
                        axs[1].plot(T, np.asarray(data['specific_heat'])[sort_idx])
                        axs[1].set(xlabel="Temperature", ylabel="Specific Heat", xscale='log')
                        axs[1].grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(thermo_plots_dir,
                                                 f"mTPQ_thermo_cluster_{cluster_id}_order_{order}.png"))
                        plt.close(fig)
                    else:
                        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                        fig.suptitle(f"Thermodynamics: cluster {cluster_id} (order {order})")
                        for ax, key, color, label in [
                            (axs[0, 0], 'energy', 'r-', "Energy"),
                            (axs[0, 1], 'specific_heat', 'b-', "Specific Heat"),
                            (axs[1, 0], 'entropy', 'g-', "Entropy"),
                            (axs[1, 1], 'free_energy', 'm-', "Free Energy"),
                        ]:
                            if data.get(key) is not None:
                                ax.plot(T, np.asarray(data[key])[sort_idx], color)
                            ax.set(xlabel="Temperature", ylabel=label, xscale='log')
                            ax.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(thermo_plots_dir,
                                                 f"thermo_cluster_{cluster_id}_order_{order}.png"))
                        plt.close(fig)
                except Exception as e:  # pragma: no cover - plot-only failure
                    logging.error(f"Error plotting cluster {cluster_id}: {e}")
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
            # NLC_sum_TPQ.py never made it into the repository; the mTPQ
            # branch was a placeholder. Fail loudly rather than silently
            # launching a missing script.
            raise NotImplementedError(
                "NLCE summation for mTPQ is not implemented. "
                "Use --method=FULL (NLC_sum.py), --method=FTLM "
                "(nlce_ftlm.py + NLC_sum_ftlm.py), or --lanczos-boost "
                "(NLC_sum_LB.py) instead."
            )
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