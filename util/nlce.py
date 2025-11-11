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
    """Run ED for a single cluster"""
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
    elif ed_options["method"] == 'FULL' or ed_options["method"] == 'OSS':
        cmd = [
            ed_executable,
            ham_subdir,
            f'--method={ed_options["method"]}',
            f'--eigenvalues=FULL',
            f'--output={cluster_ed_dir}/output',
            f'--num_sites={num_sites}',
            '--spin_length=0.5'
        ]

    if ed_options["measure_spin"]:
        cmd.append('--measure_spin')

    if symmetrized:
        cmd.append('--symmetrized')
    
    # Add thermodynamic parameters if required
    if ed_options["thermo"]:
        cmd.extend([
            '--thermo',
            f'--temp_min={ed_options["temp_min"]}',
            f'--temp_max={ed_options["temp_max"]}',
            f'--temp_bins={ed_options["temp_bins"]}'
        ])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        # Check if the computation actually succeeded despite the error
        # This can happen when the ED program completes successfully but crashes during cleanup
        expected_output_dir = os.path.join(cluster_ed_dir, 'output')
        
        # Check if output directory exists and has content
        if os.path.exists(expected_output_dir):
            output_files = os.listdir(expected_output_dir)
            
            if ed_options["thermo"]:
                # For thermodynamic calculations, check for thermo directory
                thermo_dir = os.path.join(expected_output_dir, 'thermo')
                if os.path.exists(thermo_dir) and os.listdir(thermo_dir):
                    logging.warning(f"ED for cluster {cluster_id} crashed with exit code {e.returncode} but thermodynamic output files exist - treating as success")
                    return True
            
            # For other methods, check for any meaningful output files
            if output_files and any(f.endswith(('.dat', '.txt')) or os.path.isdir(os.path.join(expected_output_dir, f)) for f in output_files):
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Automate NLCE workflow for pyrochlore lattice')
    
    # Parameters for the entire workflow
    parser.add_argument('--max_order', type=int, required=True, help='Maximum order of clusters to generate')
    parser.add_argument('--base_dir', type=str, default='./nlce_results', help='Base directory for all results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED', help='Path to the ED executable')
    
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

    parser.add_argument('--symmetrized', action='store_true', help='Use symmetrized Hamiltonian')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')

    # Random transverse field
    parser.add_argument('--random_field_width', type=float, default=0, help='Width of the random transverse field')

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
            'python3', 
            os.path.join(os.path.dirname(__file__), 'generate_pyrochlore_clusters.py'),
            f'--max_order={args.max_order}',
            f'--output_dir={cluster_dir}'
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
                logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
                logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
    else:
        logging.info("Skipping Hamiltonian preparation step.")
    
    # Step 3: Run Exact Diagonalization for each cluster
    if not args.skip_ed:
        logging.info("="*80)
        logging.info("Step 3: Running Exact Diagonalization for each cluster")
        logging.info("="*80)
        
        # Prepare ED options
        ed_options = {
            "method": args.method,
            "thermo": args.thermo,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "temp_bins": args.temp_bins,
            "measure_spin": args.measure_spin
        }
        
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

                # Check if thermodynamic data exists (ED writes into output/thermo)
                thermo_file = os.path.join(cluster_ed_dir, "output/thermo/thermo_data.txt")
                if not os.path.exists(thermo_file):
                    logging.warning(f"No thermodynamic data found for cluster {cluster_id}")
                    continue
                
                # Load thermodynamic data
                try:
                    # Parse header to determine the available columns
                    columns = []
                    with open(thermo_file, 'r') as f:
                        for line in f:
                            if not line.startswith('#'):
                                break
                            stripped = line.lstrip('#').strip()
                            if not stripped:
                                continue
                            if 'Column' in stripped:
                                parts = stripped.split(':')
                                if len(parts) >= 2:
                                    col_num = int(parts[0].replace('Column', '').strip()) - 1
                                    col_name = parts[1].strip()
                                    while len(columns) <= col_num:
                                        columns.append(None)
                                    columns[col_num] = col_name
                            else:
                                tokens = stripped.split()
                                if tokens:
                                    columns = tokens

                    if not columns:
                        # Fall back to the expected ordering if the header is absent
                        columns = ['Temperature', 'Energy', 'Specific Heat', 'Entropy', 'Free Energy']

                    def _normalize(name):
                        return name.strip().lower().replace('_', ' ')

                    normalized_columns = [(_normalize(col) if col else None) for col in columns]

                    def _get_index(candidates, default_idx):
                        for candidate in candidates:
                            norm_candidate = _normalize(candidate)
                            if norm_candidate in normalized_columns:
                                return normalized_columns.index(norm_candidate)
                        return default_idx

                    # Load the data
                    data = np.loadtxt(thermo_file, comments='#')
                    data = np.atleast_2d(data)

                    # Create plots
                    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle(f"Thermodynamic Properties for Cluster {cluster_id} (Order {order})")

                    temp_idx = _get_index(['Temperature', 'Temp', 'T'], 0)
                    energy_idx = _get_index(['Energy', 'Internal Energy'], 1 if data.shape[1] > 1 else 0)
                    spec_heat_idx = _get_index(['Specific Heat', 'Specific_Heat', 'SpecificHeat'], 2 if data.shape[1] > 2 else min(energy_idx + 1, data.shape[1] - 1))
                    entropy_idx = _get_index(['Entropy'], 3 if data.shape[1] > 3 else min(spec_heat_idx + 1, data.shape[1] - 1))
                    free_energy_idx = _get_index(['Free Energy', 'Free_Energy', 'FreeEnergy'], 4 if data.shape[1] > 4 else min(entropy_idx + 1, data.shape[1] - 1))

                    T = data[:, temp_idx]

                    # Sort by temperature (ascending) for smoother plots
                    sort_idx = np.argsort(T)
                    T = T[sort_idx]
                    sorted_data = data[sort_idx]

                    # Plot energy
                    axs[0, 0].plot(T, sorted_data[:, energy_idx], 'r-')
                    axs[0, 0].set_xlabel("Temperature")
                    axs[0, 0].set_ylabel("Energy")
                    axs[0, 0].set_xscale('log')
                    axs[0, 0].grid(True)

                    # Plot specific heat
                    axs[0, 1].plot(T, sorted_data[:, spec_heat_idx], 'b-')
                    axs[0, 1].set_xlabel("Temperature")
                    axs[0, 1].set_ylabel("Specific Heat")
                    axs[0, 1].set_xscale('log')
                    axs[0, 1].grid(True)

                    # Plot entropy
                    axs[1, 0].plot(T, sorted_data[:, entropy_idx], 'g-')
                    axs[1, 0].set_xlabel("Temperature")
                    axs[1, 0].set_ylabel("Entropy")
                    axs[1, 0].set_xscale('log')
                    axs[1, 0].grid(True)

                    # Plot free energy
                    axs[1, 1].plot(T, sorted_data[:, free_energy_idx], 'm-')
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
                    
                # Find all SS_rand*.dat files
                ss_files = os.path.join(output_dir, "SS_rand0.dat")
                
                if not ss_files:
                    logging.warning(f"No SS_rand*.dat files found for cluster {cluster_id}")
                    continue
                    
                SS_data = np.loadtxt(ss_files, unpack=True, skiprows=2)

                logging.info(f"Loaded data from {ss_files} for cluster {cluster_id}")   

                all_temps = 1.0 / SS_data[0]
                all_energies = SS_data[1] 
                all_variances = SS_data[2] * SS_data[0]**2

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
        
        if args.method == 'mTPQ':
            logging.info("Using mTPQ method for NLCE summation")
            # Add mTPQ specific parameters here if needed
            nlc_params = [
                'python3',
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
                'python3',
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