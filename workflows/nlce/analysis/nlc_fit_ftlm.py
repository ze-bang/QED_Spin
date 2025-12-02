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
import shutil
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
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


def load_multiple_experimental_data(exp_data_configs):
    """Load experimental data from multiple files with their respective parameters"""
    exp_datasets = []
    for config in exp_data_configs:
        temp, spec_heat = load_experimental_data(config['file'])
        dataset = {
            'temp': temp,
            'spec_heat': spec_heat,
            'h': config['h'],
            'field_dir': config.get('field_dir', [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
            'weight': config.get('weight', 1.0)
        }
        # Add temp_min and temp_max if they exist in the config
        if 'temp_min' in config:
            dataset['temp_min'] = config['temp_min']
        if 'temp_max' in config:
            dataset['temp_max'] = config['temp_max']
        exp_datasets.append(dataset)
    return exp_datasets


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
    h_value = h_field * 5.4 * 0.0578 if h_field is not None else fixed_params["h"]
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
    
    # Check if results already exist (use correct directory name with order)
    max_order = fixed_params["max_order"]
    nlc_result_file = os.path.join(run_dir, f'nlc_results_order_{max_order}', 'nlc_specific_heat.txt')
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
            # Consistent normalization: divide by 8 (number of sites per unit cell for pyrochlore)
            return calc_temp, interp_spec_heat / 8
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
        '--SI_units'
    ]
    
    # Add ED executable path if specified
    if 'ed_executable' in fixed_params and fixed_params['ed_executable']:
        cmd.extend(['--ed_executable', fixed_params['ed_executable']])
    
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
            
            return calc_temp, interp_spec_heat/8
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


def save_checkpoint(params, chi_squared, iteration, output_dir):
    """Save current optimization state to checkpoint file"""
    checkpoint = {
        'params': params.tolist() if isinstance(params, np.ndarray) else params,
        'chi_squared': float(chi_squared),
        'iteration': int(iteration),
        'timestamp': np.datetime64('now').astype(str)
    }
    
    checkpoint_file = os.path.join(output_dir, 'fit_checkpoint.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    logging.debug(f"Checkpoint saved: iteration {iteration}, chi_squared={chi_squared:.6f}")


def load_checkpoint(output_dir):
    """Load optimization state from checkpoint file"""
    checkpoint_file = os.path.join(output_dir, 'fit_checkpoint.json')
    
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        params = np.array(checkpoint['params'])
        logging.info(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
        logging.info(f"  Parameters: Jxx={params[0]:.6f}, Jyy={params[1]:.6f}, Jzz={params[2]:.6f}")
        logging.info(f"  Chi-squared: {checkpoint['chi_squared']:.6f}")
        logging.info(f"  Timestamp: {checkpoint['timestamp']}")
        
        return params
    except Exception as e:
        logging.warning(f"Failed to load checkpoint: {e}")
        return None


def calc_chi_squared(params, fixed_params, exp_datasets, work_dir, output_dir=None):
    """Calculate combined chi-squared between experimental and calculated specific heat for multiple datasets"""
    total_chi_squared = 0.0
    
    # Clear cache if requested
    if fixed_params.get("clear_cache", False):
        cache_dirs = glob.glob(os.path.join(work_dir, 'run_*'))
        for cache_dir in cache_dirs:
            try:
                shutil.rmtree(cache_dir)
                logging.debug(f"Cleared cache directory: {cache_dir}")
            except Exception as e:
                logging.debug(f"Failed to clear cache {cache_dir}: {e}")
    
    for i, dataset in enumerate(exp_datasets):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset['h']
        weight = dataset['weight']
        
        # Get dataset-specific temperature range if available
        temp_range = None
        if 'temp_min' in dataset and 'temp_max' in dataset:
            temp_range = (dataset['temp_min'], dataset['temp_max'])
        
        # Run NLCE with dataset-specific h field
        calc_temp, calc_spec_heat = run_nlce_ftlm(params, fixed_params, exp_temp, work_dir, 
                                                   h_field=h_field, temp_range=temp_range)
        
        if len(calc_spec_heat) == 0:
            logging.error(f"No calculated data available for dataset {i+1}")
            return 1e10  # Large penalty
        
        # Calculate chi-squared for this dataset
        diff = exp_spec_heat - calc_spec_heat
        dataset_chi_squared = np.sum(diff**2)
        
        # Apply dataset weight
        weighted_chi_squared = weight * dataset_chi_squared
        total_chi_squared += weighted_chi_squared
        
        logging.info(f"Dataset {i+1} (h={h_field}): Chi-squared={dataset_chi_squared:.4f}, "
                    f"Weighted={weighted_chi_squared:.4f}")
    
    logging.info(f"Parameters: Jxx={params[0]:.4f}, Jyy={params[1]:.4f}, Jzz={params[2]:.4f}, "
                f"Total Chi-squared={total_chi_squared:.4f}")
    
    # Save checkpoint if output_dir is provided
    if output_dir is not None:
        if not hasattr(calc_chi_squared, 'iteration'):
            calc_chi_squared.iteration = 0
        calc_chi_squared.iteration += 1
        save_checkpoint(params, total_chi_squared, calc_chi_squared.iteration, output_dir)
    
    return total_chi_squared


def plot_results(exp_datasets, best_params, fixed_params, work_dir, output_dir):
    """Plot experimental data and best fit for all datasets (without error bars)"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("Matplotlib not available. Skipping plot.")
        return
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_datasets)))
    
    for i, (dataset, color) in enumerate(zip(exp_datasets, colors)):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset['h']
        
        # Get dataset-specific temperature range if available
        temp_range = None
        if 'temp_min' in dataset and 'temp_max' in dataset:
            temp_range = (dataset['temp_min'], dataset['temp_max'])
        
        # Plot experimental data (without error bars)
        plt.scatter(exp_temp, exp_spec_heat, color=color, alpha=0.7, 
                   label=f'Experimental Data (h={h_field})', zorder=5, s=50)
        
        # Calculate and plot fitted data
        calc_temp, calc_spec_heat = run_nlce_ftlm(best_params, fixed_params, exp_temp, work_dir,
                                                   h_field=h_field, temp_range=temp_range)
        
        if len(calc_temp) > 0:
            # Sort calculated data by temperature
            sort_idx = np.argsort(calc_temp)
            calc_temp = calc_temp[sort_idx]
            calc_spec_heat = calc_spec_heat[sort_idx]
            
            # Plot calculated data
            plt.plot(calc_temp, calc_spec_heat, color=color, linestyle='-', linewidth=2,
                    label=f'NLCE-FTLM Fit (h={h_field})')
    
    title_str = f'NLCE-FTLM Fit: Jxx={best_params[0]:.3f}, Jyy={best_params[1]:.3f}, Jzz={best_params[2]:.3f}'
    
    plt.xlabel('Temperature (K)', fontsize=12)
    plt.ylabel('Specific Heat (J/mol·K)', fontsize=12)
    plt.title(title_str, fontsize=14)
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
  # Single dataset fitting
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --output_dir fit_results/
  
  # Resume from checkpoint
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --output_dir fit_results/ --resume
  
  # Load from specific checkpoint file
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --checkpoint_file path/to/fit_checkpoint.json
  
  # Multiple datasets with config file
  python nlc_fit_ftlm.py --exp_config exp_config.json --max_order 4 --output_dir fit_results/
  
  # With custom FTLM parameters
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --ftlm_samples 30 --krylov_dim 200
  
  # Parallel execution
  python nlc_fit_ftlm.py --exp_data experimental_cv.txt --max_order 4 --parallel --num_cores 8

Example config file (exp_config.json):
{
  "experimental_data": [
    {
      "file": "data_h0.txt",
      "h": 0.0,
      "weight": 1.0,
      "temp_min": 0.1,
      "temp_max": 10.0
    },
    {
      "file": "data_h1.txt",
      "h": 1.0,
      "field_dir": [0.5773, 0.5773, 0.5773],
      "weight": 1.0,
      "temp_min": 0.1,
      "temp_max": 10.0
    }
  ],
  "global_params": {
    "max_order": 4,
    "ftlm_samples": 30
  }
}
        """
    )
    
    # Input/output
    parser.add_argument('--exp_data', type=str, default='specific_heat_data.txt',
                       help='Path to experimental specific heat data (for single file mode)')
    parser.add_argument('--exp_config', type=str, default=None,
                       help='Path to JSON config file for multiple experimental datasets')
    parser.add_argument('--output_dir', default='./nlce_ftlm_fit', help='Output directory')
    parser.add_argument('--work_dir', default='./nlce_ftlm_fit_work', help='Working directory')
    parser.add_argument('--ed_executable', type=str, default='./build/FTLM',
                       help='Path to the FTLM executable')
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=4, help='Maximum cluster order')
    
    # FTLM parameters
    parser.add_argument('--ftlm_samples', type=int, default=100, help='Number of FTLM samples')
    parser.add_argument('--krylov_dim', type=int, default=150, help='Krylov subspace dimension')
    
    # Model parameters (initial guess)
    parser.add_argument('--Jxx_init', type=float, default=1.0, help='Initial Jxx')
    parser.add_argument('--Jyy_init', type=float, default=1.0, help='Initial Jyy')
    parser.add_argument('--Jzz_init', type=float, default=1.0, help='Initial Jzz')
    
    # Parameter bounds
    parser.add_argument('--Jxx_bounds', type=float, nargs=2, default=[-4.0, 4.0])
    parser.add_argument('--Jyy_bounds', type=float, nargs=2, default=[0, 8.0])
    parser.add_argument('--Jzz_bounds', type=float, nargs=2, default=[0.1, 8.0])
    
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
                       choices=['Nelder-Mead', 'Powell', 'COBYLA', 'L-BFGS-B', 'SLSQP', 'trust-constr'],
                       help='Optimization method')
    parser.add_argument('--maxiter', type=int, default=50, help='Maximum iterations')
    parser.add_argument('--timeout', type=int, default=3600, 
                       help='Timeout for each NLCE-FTLM run (seconds)')
    
    # Execution
    parser.add_argument('--parallel', action='store_true', help='Run FTLM in parallel')
    parser.add_argument('--num_cores', type=int, default=4, help='Number of cores for parallel')
    parser.add_argument('--symmetrized', action='store_true', help='Use symmetrized Hamiltonian')
    parser.add_argument('--clear_cache', action='store_true', 
                       help='Clear cached results between optimization steps')
    
    # Checkpoint functionality
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from checkpoint if available')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                       help='Path to specific checkpoint file to load (overrides --resume)')
    
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
    
    # Load experimental data
    if args.exp_config:
        # Load multiple datasets from config file
        logging.info(f"Loading experimental data configuration from {args.exp_config}")
        with open(args.exp_config, 'r') as f:
            config = json.load(f)
        
        exp_datasets = load_multiple_experimental_data(config['experimental_data'])
        logging.info(f"Loaded {len(exp_datasets)} experimental datasets")
        
        # Update fixed_params with any global config values
        if 'global_params' in config:
            for key, value in config['global_params'].items():
                if key in ['max_order', 'ftlm_samples', 'krylov_dim', 'temp_min', 
                          'temp_max', 'temp_bins', 'h', 'field_dir']:
                    logging.info(f"Overriding {key} from config: {value}")
    else:
        # Single dataset mode (backward compatibility)
        logging.info(f"Loading experimental data from {args.exp_data}")
        exp_temp, exp_spec_heat = load_experimental_data(args.exp_data)
        
        # Create single dataset
        exp_datasets = [{
            'temp': exp_temp,
            'spec_heat': exp_spec_heat,
            'h': args.h,
            'field_dir': args.field_dir,
            'weight': 1.0
        }]
        logging.info(f"Loaded {len(exp_temp)} data points")
    
    logging.info(f"Max order: {args.max_order}")
    logging.info(f"FTLM samples: {args.ftlm_samples}")
    logging.info(f"Krylov dimension: {args.krylov_dim}")
    logging.info("="*80)
    
    # Filter experimental data based on temperature range for all datasets
    filtered_datasets = []
    for dataset in exp_datasets:
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        
        # Use dataset-specific temp range if available, otherwise use global
        temp_min = dataset.get('temp_min', args.temp_min)
        temp_max = dataset.get('temp_max', args.temp_max)
        
        valid_indices = (exp_temp >= temp_min) & (exp_temp <= temp_max)
        filtered_temp = exp_temp[valid_indices]
        filtered_spec_heat = exp_spec_heat[valid_indices]
        
        if len(filtered_temp) > 0:
            filtered_dataset = dataset.copy()
            filtered_dataset['temp'] = filtered_temp
            filtered_dataset['spec_heat'] = filtered_spec_heat
            filtered_datasets.append(filtered_dataset)
            logging.info(f"Dataset h={dataset['h']}: {len(filtered_temp)} points in "
                        f"range [{temp_min:.4f}, {temp_max:.4f}] K")
        else:
            logging.warning(f"Dataset h={dataset['h']}: No points in temperature range")
    
    if not filtered_datasets:
        logging.error("No experimental data points within temperature range for any dataset")
        sys.exit(1)
    
    exp_datasets = filtered_datasets
    
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
        'timeout': args.timeout,
        'ed_executable': args.ed_executable,
        'clear_cache': args.clear_cache
    }
    
    # Initial parameters and bounds
    initial_params = np.array([args.Jxx_init, args.Jyy_init, args.Jzz_init])
    
    # Try to load from checkpoint if requested
    if args.checkpoint_file:
        # Load from specific checkpoint file
        logging.info(f"Loading checkpoint from: {args.checkpoint_file}")
        try:
            with open(args.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            loaded_params = np.array(checkpoint['params'])
            logging.info(f"Loaded parameters from checkpoint: Jxx={loaded_params[0]:.6f}, "
                        f"Jyy={loaded_params[1]:.6f}, Jzz={loaded_params[2]:.6f}")
            initial_params = loaded_params
        except Exception as e:
            logging.warning(f"Failed to load checkpoint file {args.checkpoint_file}: {e}")
            logging.info("Using default initial parameters")
    elif args.resume:
        # Try to load from default checkpoint in output_dir
        loaded_params = load_checkpoint(args.output_dir)
        if loaded_params is not None:
            initial_params = loaded_params
        else:
            logging.info("No checkpoint found, using default initial parameters")
    
    bounds = [
        tuple(args.Jxx_bounds),
        tuple(args.Jyy_bounds),
        tuple(args.Jzz_bounds)
    ]
    
    logging.info(f"Starting parameters: Jxx={initial_params[0]:.4f}, "
                f"Jyy={initial_params[1]:.4f}, Jzz={initial_params[2]:.4f}")
    logging.info(f"Parameter bounds: Jxx={bounds[0]}, Jyy={bounds[1]}, Jzz={bounds[2]}")
    
    # Define constraints
    def constraint_func(params):
        Jxx, Jyy, Jzz = params[:3]
        # Return array of constraint values (should be >= 0)
        return np.array([
            0.125*Jzz - Jxx,
            0.2*Jzz - Jyy + 0.4 * Jxx,
            Jyy + 0.4 * Jxx + 0.2*Jzz,
        ])
    
    constraints = NonlinearConstraint(constraint_func, 0, np.inf)
    
    # Run optimization
    logging.info(f"\nStarting optimization with method: {args.method}")
    logging.info(f"Maximum iterations: {args.maxiter}")
    logging.info("="*80)
    
    # Reset iteration counter for checkpoint saving
    calc_chi_squared.iteration = 0
    
    result = minimize(
        calc_chi_squared,
        initial_params,
        args=(fixed_params, exp_datasets, args.work_dir, args.output_dir),
        method=args.method,
        bounds=bounds if args.method == 'L-BFGS-B' else None,
        constraints=constraints if args.method in ['SLSQP', 'COBYLA', 'trust-constr'] else None,
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
    plot_results(exp_datasets, result.x, fixed_params, 
                args.work_dir, args.output_dir)
    
    logging.info(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time)/60:.2f} minutes")
