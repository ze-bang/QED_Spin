#!/usr/bin/env python3
"""
NLCE Fitting Tool for FTLM-based calculations

This script fits Numerical Linked Cluster Expansion (NLCE) calculations using
FTLM (Finite Temperature Lanczos Method) to experimental specific heat data.

Key differences from nlc_fit.py:
- Uses FTLM instead of full diagonalization for each cluster
- FTLM provides direct thermodynamic quantities with error bars
- Scales to larger clusters than full diagonalization
- Faster iterations during fitting
"""

import os
import sys
import glob
import re
import logging
import argparse
import subprocess
import json
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d


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


def load_experimental_data(file_path):
    """Load experimental data from file"""
    data = np.loadtxt(file_path)
    temp = data[:, 0]  # Temperature (K)
    spec_heat = data[:, 1]  # Specific heat (J*mol/K)
    return temp, spec_heat


def run_nlce_ftlm(params, fixed_params, exp_temp, work_dir, h_field=None, temp_range=None):
    """
    Run NLCE with FTLM for the given parameters and return calculated specific heat
    
    Args:
        params: Array of [Jxx, Jyy, Jzz] coupling parameters
        fixed_params: Dictionary of fixed parameters
        exp_temp: Experimental temperature grid
        work_dir: Working directory for calculations
        h_field: Optional field value override
        temp_range: Optional (temp_min, temp_max) override
        
    Returns:
        Tuple of (calc_temp, calc_spec_heat) interpolated to exp_temp grid
    """
    logging.info(f"Running NLCE-FTLM with J=({params[0]:.4f}, {params[1]:.4f}, {params[2]:.4f})")
    
    Jxx, Jyy, Jzz = params[:3]
    h_value = h_field if h_field is not None else fixed_params["h"]
    field_dir = fixed_params["field_dir"]
    
    # Temperature range
    if temp_range is not None:
        temp_min, temp_max = temp_range
    else:
        temp_min = fixed_params["temp_min"]
        temp_max = fixed_params["temp_max"]
    
    # Create unique subdirectory for this run
    import hashlib
    param_hash = hashlib.md5(f"{Jxx}{Jyy}{Jzz}{h_value}".encode()).hexdigest()[:8]
    run_dir = os.path.join(work_dir, f'run_{param_hash}')
    
    # Check if results already exist
    nlc_result_file = os.path.join(run_dir, 'nlc_results', 'nlc_specific_heat.txt')
    if os.path.exists(nlc_result_file):
        logging.info(f"Using cached results from {run_dir}")
        try:
            data = np.loadtxt(nlc_result_file)
            calc_temp = data[:, 0]
            calc_spec_heat = data[:, 1]
            
            # Interpolate to experimental grid
            interp_func = interp1d(calc_temp, calc_spec_heat, kind='cubic',
                                  bounds_error=False, fill_value=0.0)
            interp_spec_heat = interp_func(exp_temp)
            return calc_temp, interp_spec_heat
        except Exception as e:
            logging.warning(f"Failed to load cached results: {e}")
    
    os.makedirs(run_dir, exist_ok=True)
    
    # Build NLCE-FTLM command
    cmd = [
        'python3',
        os.path.join(os.path.dirname(__file__), 'nlce_ftlm.py'),
        f'--max_order={fixed_params["max_order"]}',
        f'--base_dir={run_dir}',
        f'--Jxx={Jxx}',
        f'--Jyy={Jyy}',
        f'--Jzz={Jzz}',
        f'--h={h_value}',
        f'--field_dir', str(field_dir[0]), str(field_dir[1]), str(field_dir[2]),
        f'--ftlm_samples={fixed_params["ftlm_samples"]}',
        f'--krylov_dim={fixed_params["krylov_dim"]}',
        f'--temp_min={temp_min}',
        f'--temp_max={temp_max}',
        f'--temp_bins={fixed_params["temp_bins"]}',
    ]
    
    if fixed_params.get("symmetrized", False):
        cmd.append('--symmetrized')
    
    if fixed_params.get("parallel", False):
        cmd.append('--parallel')
        cmd.append(f'--num_cores={fixed_params.get("num_cores", 4)}')
    
    # Run NLCE-FTLM
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                              timeout=fixed_params.get("timeout", 3600))
        
        # Load results
        if os.path.exists(nlc_result_file):
            data = np.loadtxt(nlc_result_file)
            calc_temp = data[:, 0]
            calc_spec_heat = data[:, 1]
            
            # Interpolate to experimental grid
            interp_func = interp1d(calc_temp, calc_spec_heat, kind='cubic',
                                  bounds_error=False, fill_value=0.0)
            interp_spec_heat = interp_func(exp_temp)
            
            return calc_temp, interp_spec_heat
        else:
            logging.error(f"NLC results file not found: {nlc_result_file}")
            return np.array([]), np.zeros_like(exp_temp)
            
    except subprocess.TimeoutExpired:
        logging.error(f"NLCE-FTLM calculation timed out after {fixed_params.get('timeout', 3600)}s")
        return np.array([]), np.zeros_like(exp_temp)
    except subprocess.CalledProcessError as e:
        logging.error(f"NLCE-FTLM calculation failed: {e}")
        if e.stdout:
            logging.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"Stderr: {e.stderr}")
        return np.array([]), np.zeros_like(exp_temp)


def calc_chi_squared(params, fixed_params, exp_temp, exp_spec_heat, work_dir):
    """Calculate chi-squared between experimental and calculated specific heat"""
    
    _, calc_spec_heat = run_nlce_ftlm(params, fixed_params, exp_temp, work_dir)
    
    if len(calc_spec_heat) == 0:
        logging.error("No calculated data available")
        return 1e10  # Large penalty
    
    # Calculate chi-squared
    diff = exp_spec_heat - calc_spec_heat
    chi_squared = np.sum(diff**2)
    
    logging.info(f"Parameters: Jxx={params[0]:.4f}, Jyy={params[1]:.4f}, Jzz={params[2]:.4f}, "
                f"Chi-squared={chi_squared:.4f}")
    
    return chi_squared


def plot_results(exp_temp, exp_spec_heat, best_params, fixed_params, work_dir, output_dir):
    """Plot experimental data and best fit"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("Matplotlib not available. Skipping plot.")
        return
    
    # Get best fit results
    _, best_spec_heat = run_nlce_ftlm(best_params, fixed_params, exp_temp, work_dir)
    
    plt.figure(figsize=(10, 6))
    plt.plot(exp_temp, exp_spec_heat, 'o', label='Experimental', markersize=6)
    plt.plot(exp_temp, best_spec_heat, '-', linewidth=2, 
            label=f'NLCE-FTLM Fit: Jxx={best_params[0]:.3f}, Jyy={best_params[1]:.3f}, Jzz={best_params[2]:.3f}')
    
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Specific Heat (J/mol·K)', fontsize=12)
    plt.title('NLCE-FTLM Fit to Experimental Data', fontsize=14)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'nlce_ftlm_fit.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Fit NLCE-FTLM calculations to experimental specific heat data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic fitting
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --output_dir fit_results/
  
  # With custom FTLM parameters
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --ftlm_samples 30 --krylov_dim 200
  
  # Parallel execution
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --parallel --num_cores 8
        """
    )
    
    # Input/output
    parser.add_argument('--exp_data', required=True, help='Experimental specific heat data file')
    parser.add_argument('--output_dir', default='./nlce_ftlm_fit', help='Output directory')
    parser.add_argument('--work_dir', default='./nlce_ftlm_fit_work', help='Working directory')
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=4, help='Maximum cluster order')
    
    # FTLM parameters
    parser.add_argument('--ftlm_samples', type=int, default=20, help='Number of FTLM samples')
    parser.add_argument('--krylov_dim', type=int, default=150, help='Krylov subspace dimension')
    
    # Model parameters (initial guess)
    parser.add_argument('--Jxx_init', type=float, default=1.0, help='Initial Jxx')
    parser.add_argument('--Jyy_init', type=float, default=1.0, help='Initial Jyy')
    parser.add_argument('--Jzz_init', type=float, default=1.0, help='Initial Jzz')
    
    # Parameter bounds
    parser.add_argument('--Jxx_bounds', type=float, nargs=2, default=[0.1, 2.0])
    parser.add_argument('--Jyy_bounds', type=float, nargs=2, default=[0.1, 2.0])
    parser.add_argument('--Jzz_bounds', type=float, nargs=2, default=[0.1, 2.0])
    
    # Other model parameters
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field')
    parser.add_argument('--field_dir', type=float, nargs=3,
                       default=[1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    
    # Temperature range
    parser.add_argument('--temp_min', type=float, default=0.001, help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=20.0, help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100, help='Number of temperature bins')
    
    # Optimization
    parser.add_argument('--method', default='Nelder-Mead', 
                       choices=['Nelder-Mead', 'Powell', 'COBYLA', 'L-BFGS-B'],
                       help='Optimization method')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximum iterations')
    parser.add_argument('--timeout', type=int, default=3600, 
                       help='Timeout for each NLCE-FTLM run (seconds)')
    
    # Execution
    parser.add_argument('--parallel', action='store_true', help='Run FTLM in parallel')
    parser.add_argument('--num_cores', type=int, default=4, help='Number of cores for parallel')
    parser.add_argument('--symmetrized', action='store_true', help='Use symmetrized Hamiltonian')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'nlce_ftlm_fit.log')
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info("NLCE-FTLM Fitting Tool")
    logging.info("="*80)
    logging.info(f"Experimental data: {args.exp_data}")
    logging.info(f"Max order: {args.max_order}")
    logging.info(f"FTLM samples: {args.ftlm_samples}")
    logging.info(f"Krylov dimension: {args.krylov_dim}")
    logging.info("="*80)
    
    # Load experimental data
    exp_temp, exp_spec_heat = load_experimental_data(args.exp_data)
    logging.info(f"Loaded {len(exp_temp)} experimental data points")
    logging.info(f"Temperature range: [{exp_temp.min():.4f}, {exp_temp.max():.4f}] K")
    
    # Set up fixed parameters
    fixed_params = {
        'max_order': args.max_order,
        'ftlm_samples': args.ftlm_samples,
        'krylov_dim': args.krylov_dim,
        'temp_min': args.temp_min,
        'temp_max': args.temp_max,
        'temp_bins': args.temp_bins,
        'h': args.h,
        'field_dir': args.field_dir,
        'symmetrized': args.symmetrized,
        'parallel': args.parallel,
        'num_cores': args.num_cores,
        'timeout': args.timeout
    }
    
    # Initial parameters and bounds
    initial_params = np.array([args.Jxx_init, args.Jyy_init, args.Jzz_init])
    bounds = [
        tuple(args.Jxx_bounds),
        tuple(args.Jyy_bounds),
        tuple(args.Jzz_bounds)
    ]
    
    logging.info(f"Initial parameters: Jxx={initial_params[0]:.4f}, "
                f"Jyy={initial_params[1]:.4f}, Jzz={initial_params[2]:.4f}")
    logging.info(f"Parameter bounds: Jxx={bounds[0]}, Jyy={bounds[1]}, Jzz={bounds[2]}")
    
    # Run optimization
    logging.info(f"\nStarting optimization with method: {args.method}")
    logging.info(f"Maximum iterations: {args.maxiter}")
    logging.info("="*80)
    
    result = minimize(
        calc_chi_squared,
        initial_params,
        args=(fixed_params, exp_temp, exp_spec_heat, args.work_dir),
        method=args.method,
        bounds=bounds if args.method == 'L-BFGS-B' else None,
        options={'maxiter': args.maxiter, 'disp': True}
    )
    
    # Report results
    logging.info("="*80)
    logging.info("Optimization completed")
    logging.info("="*80)
    logging.info(f"Success: {result.success}")
    logging.info(f"Message: {result.message}")
    logging.info(f"Iterations: {result.nit}")
    logging.info(f"Function evaluations: {result.nfev}")
    logging.info(f"\nBest parameters:")
    logging.info(f"  Jxx = {result.x[0]:.6f}")
    logging.info(f"  Jyy = {result.x[1]:.6f}")
    logging.info(f"  Jzz = {result.x[2]:.6f}")
    logging.info(f"\nChi-squared: {result.fun:.6f}")
    logging.info("="*80)
    
    # Save results
    results_dict = {
        'parameters': {
            'Jxx': float(result.x[0]),
            'Jyy': float(result.x[1]),
            'Jzz': float(result.x[2])
        },
        'chi_squared': float(result.fun),
        'success': bool(result.success),
        'iterations': int(result.nit),
        'function_evaluations': int(result.nfev),
        'message': str(result.message),
        'fixed_params': {k: (v if not isinstance(v, np.ndarray) else v.tolist()) 
                        for k, v in fixed_params.items()}
    }
    
    results_file = os.path.join(args.output_dir, 'fit_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logging.info(f"Results saved to: {results_file}")
    
    # Generate plot
    plot_results(exp_temp, exp_spec_heat, result.x, fixed_params, 
                args.work_dir, args.output_dir)
    
    logging.info(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
