import os
import sys
import subprocess
import argparse
import time
import logging
import numpy as np
import multiprocessing
from tqdm import tqdm

#!/usr/bin/env python3
"""
Pyrochlore Magnetic Field Scan

This script performs a scan of different magnetic field strengths 
on a pyrochlore lattice. For each field strength:
1. Creates a directory
2. Generates the pyrochlore lattice using helper_pyrochlore.py
3. Runs Exact Diagonalization
4. Analyzes results (energies, magnetization, etc.)
"""

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

def generate_pyrochlore(args, h_value, field_dir, output_dir):
    """Generate pyrochlore lattice with specified field strength"""
    cmd = [
        'python3',
        'util/helper_pyrochlore.py',
        str(args.Jxx),
        str(args.Jyy),
        str(args.Jzz),
        str(h_value),
        str(field_dir[0]),
        str(field_dir[1]),
        str(field_dir[2]),
        output_dir,
        str(args.dim1),
        str(args.dim2),
        str(args.dim3),
        str(int(args.pbc))
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error generating pyrochlore at h={h_value}: {e}")
        logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return False

def run_ed(args, h_value, output_dir):
    """Run Exact Diagonalization for the specified configuration"""
    # Count number of sites from the site_info file
    site_info_files = os.listdir(output_dir)
    site_info_file = None
    for file in site_info_files:
        if file.endswith("_site_info.dat"):
            site_info_file = os.path.join(output_dir, file)
            break
            
    if not site_info_file:
        logging.error(f"Site info file not found in {output_dir}")
        return False
        
    # Count lines in site info file (excluding header lines)
    num_sites = 0
    with open(site_info_file, 'r') as f:
        for line in f:
            if not line.startswith('#') and line.strip():
                num_sites += 1
    
    # Prepare ED command
    cmd = [
        args.ed_executable,
        output_dir,
        f'--method={args.method}',
        f'--eigenvalues={args.eigenvalues}',
        f'--output={output_dir}/output',
        f'--num_sites={num_sites}',
        '--spin_length=0.5',
    ]
    
    if args.measure_spin:
        cmd.append('--measure_spin')
    
    if args.symmetrized:
        cmd.append('--symmetrized')
    
    if args.thermo:
        cmd.extend([
            '--thermo',
            f'--temp-min={args.temp_min}',
            f'--temp-max={args.temp_max}',
            f'--temp-bins={args.temp_bins}'
        ])
        
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running ED at h={h_value}: {e}")
        logging.error(f"Stdout: {e.stdout.decode('utf-8')}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return False

def process_mag_scan_task(args):
    """Process one field strength value"""
    h_value, base_dir, args = args
    
    # Create directory for this field strength
    h_dir = os.path.join(base_dir, f"h_{h_value:.6f}")
    os.makedirs(h_dir, exist_ok=True)
    
    # Normalize field direction
    field_dir = np.array(args.field_dir)
    field_dir = field_dir / np.linalg.norm(field_dir)
    
    # Generate pyrochlore lattice
    success = generate_pyrochlore(args, h_value, field_dir, h_dir)
    if not success:
        return False, h_value
    
    # Run ED
    success = run_ed(args, h_value, h_dir)
    if not success:
        return False, h_value
    
    return True, h_value

def analyze_results(args, base_dir, h_values):
    """Analyze results from the field scan"""
    try:
        # Initialize arrays for results
        energies = np.zeros(len(h_values))
        magnetizations = np.zeros(len(h_values))
        
        # Process each field strength
        for i, h in enumerate(h_values):
            h_dir = os.path.join(base_dir, f"h_{h:.6f}")
            
            # Read energies
            energy_file = os.path.join(h_dir, "output_energy.dat")
            if os.path.exists(energy_file):
                try:
                    energies[i] = np.loadtxt(energy_file)[0]  # Get ground state energy
                except Exception as e:
                    logging.error(f"Error reading energy for h={h}: {e}")
            
            # Read magnetization
            if args.measure_spin:
                mag_file = os.path.join(h_dir, "output_magnetization.dat")
                if os.path.exists(mag_file):
                    try:
                        magnetizations[i] = np.loadtxt(mag_file)[-1]  # Total magnetization
                    except Exception as e:
                        logging.error(f"Error reading magnetization for h={h}: {e}")
        
        # Plot results
        plt.figure(figsize=(10, 8))
        
        # Energy vs field
        plt.subplot(2, 1, 1)
        plt.plot(h_values, energies, 'o-', color='blue')
        plt.xlabel('Magnetic Field Strength (h)')
        plt.ylabel('Ground State Energy')
        plt.grid(True)
        plt.title('Energy vs Magnetic Field')
        
        # Magnetization vs field
        if args.measure_spin:
            plt.subplot(2, 1, 2)
            plt.plot(h_values, magnetizations, 'o-', color='red')
            plt.xlabel('Magnetic Field Strength (h)')
            plt.ylabel('Total Magnetization')
            plt.grid(True)
            plt.title('Magnetization vs Magnetic Field')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, "field_scan_results.png"), dpi=300)
        
        # Save data to file
        results = np.column_stack((h_values, energies, magnetizations if args.measure_spin else np.zeros_like(h_values)))
        header = "Field_Strength Energy Magnetization" if args.measure_spin else "Field_Strength Energy"
        np.savetxt(os.path.join(base_dir, "field_scan_results.txt"), results, header=header)
        
        logging.info("Analysis completed and results plotted")
    except Exception as e:
        logging.error(f"Error analyzing results: {e}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform magnetic field scan on pyrochlore lattice')
    
    # Field scan parameters
    parser.add_argument('--h_min', type=float, default=0.0, help='Minimum field strength')
    parser.add_argument('--h_max', type=float, default=10.0, help='Maximum field strength')
    parser.add_argument('--h_steps', type=int, default=21, help='Number of field strength steps')
    parser.add_argument('--h_values', type=float, nargs='+', help='Explicit field values to use (overrides h_min, h_max, h_steps)')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1], help='Field direction [x, y, z]')
    parser.add_argument('--scale', choices=['linear', 'log'], default='linear', help='Scale for field steps (linear or logarithmic)')
    
    # Directory and executable parameters
    parser.add_argument('--base_dir', type=str, default='./pyrochlore_mag_scan', help='Base directory for results')
    parser.add_argument('--ed_executable', type=str, default='./build/ED', help='Path to ED executable')
    
    # Lattice parameters
    parser.add_argument('--dim1', type=int, default=1, help='First dimension of the lattice')
    parser.add_argument('--dim2', type=int, default=1, help='Second dimension of the lattice')
    parser.add_argument('--dim3', type=int, default=1, help='Third dimension of the lattice')
    parser.add_argument('--pbc', action='store_true', help='Use periodic boundary conditions')
    
    # Model parameters
    parser.add_argument('--Jxx', type=float, default=1.0, help='Jxx coupling')
    parser.add_argument('--Jyy', type=float, default=1.0, help='Jyy coupling')
    parser.add_argument('--Jzz', type=float, default=1.0, help='Jzz coupling')
    
    # ED parameters
    parser.add_argument('--method', type=str, default='FULL', help='Diagonalization method (FULL, LANCZOS)')
    parser.add_argument('--eigenvalues', type=str, default='FULL', help='Which eigenvalues to compute (FULL, GROUND)')
    parser.add_argument('--symmetrized', action='store_true', help='Use symmetrized Hamiltonian')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')
    
    # Thermodynamic parameters
    parser.add_argument('--thermo', action='store_true', help='Compute thermodynamic properties')
    parser.add_argument('--temp_min', type=float, default=0.001, help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true', help='Run calculations in parallel')
    parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count(), help='Number of cores for parallel processing')
    
    args = parser.parse_args()
    
    # Create base directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.base_dir, 'mag_scan.log')
    setup_logging(log_file)
    
    # Determine field values
    if args.h_values:
        h_values = np.array(args.h_values)
    else:
        if args.scale == 'linear':
            h_values = np.linspace(args.h_min, args.h_max, args.h_steps)
        else:  # logarithmic
            if args.h_min <= 0:
                h_min = 1e-6  # Small positive number
                logging.warning(f"h_min set to {h_min} for logarithmic scale")
            else:
                h_min = args.h_min
            h_values = np.logspace(np.log10(h_min), np.log10(args.h_max), args.h_steps)
    
    logging.info(f"Running magnetic field scan with {len(h_values)} field values from {h_values[0]} to {h_values[-1]}")
    
    # Prepare tasks for parallel processing
    tasks = [(h, args.base_dir, args) for h in h_values]
    
    # Process tasks
    if args.parallel:
        logging.info(f"Running in parallel with {args.num_cores} cores")
        with multiprocessing.Pool(processes=args.num_cores) as pool:
            results = list(tqdm(
                pool.imap(process_mag_scan_task, tasks),
                total=len(tasks),
                desc="Processing field values"
            ))
        
        # Check results
        success_count = sum(1 for success, _ in results if success)
        logging.info(f"Successfully processed {success_count} of {len(tasks)} field values")
    else:
        # Run sequentially
        for task in tqdm(tasks, desc="Processing field values"):
            process_mag_scan_task(task)
    
    # Analyze and plot results
    analyze_results(args, args.base_dir, h_values)
    
    logging.info("="*80)
    logging.info("Magnetic field scan completed!")
    logging.info(f"Results are available in {args.base_dir}")
    logging.info("="*80)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")