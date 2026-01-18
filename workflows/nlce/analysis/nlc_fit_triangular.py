#!/usr/bin/env python3
"""
NLCE Fitting Tool for Triangular Lattice

This script fits Numerical Linked Cluster Expansion (NLCE) calculations
for triangular lattice models to experimental or target specific heat data.

Supports:
- J1-J2 Heisenberg model
- XXZ model with anisotropy
- Kitaev-Heisenberg model
- Anisotropic exchange model (YbMgGaO4-type) with bond-dependent phases:
  H = Σ_{⟨ij⟩} [J_zz S_i^z S_j^z + J_± (S_i^+ S_j^- + S_i^- S_j^+)
               + J_±± (γ_ij S_i^+ S_j^+ + γ_ij* S_i^- S_j^-)
               - i J_z±/2 ((γ_ij* S_i^+ - γ_ij S_i^-) S_j^z + h.c.)]
  where γ_ij = 1, e^{i2π/3}, e^{-i2π/3} for bonds along a1, a2, a3 directions.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import logging
import json
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.stats import qmc
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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
    """Load experimental data from the specified file"""
    data = np.loadtxt(file_path)
    temp = data[:, 0]
    spec_heat = data[:, 1]
    return temp, spec_heat


def load_multiple_experimental_data(exp_data_configs):
    """Load experimental data from multiple files"""
    exp_datasets = []
    for config in exp_data_configs:
        temp, spec_heat = load_experimental_data(config['file'])
        dataset = {
            'temp': temp,
            'spec_heat': spec_heat,
            'h': config.get('h', 0.0),
            'field_dir': config.get('field_dir', [0, 0, 1]),
            'weight': config.get('weight', 1.0)
        }
        if 'temp_min' in config:
            dataset['temp_min'] = config['temp_min']
        if 'temp_max' in config:
            dataset['temp_max'] = config['temp_max']
        exp_datasets.append(dataset)
    return exp_datasets


def apply_gaussian_broadening(temp, spec_heat, sigma, broadening_type='linear'):
    """Apply Gaussian broadening to specific heat data"""
    if sigma <= 0:
        return spec_heat
    
    if broadening_type == 'log':
        log_temp = np.log(temp)
        dt = np.mean(np.diff(log_temp))
        sigma_pixels = sigma / dt
        broadened = gaussian_filter1d(spec_heat, sigma=sigma_pixels, mode='nearest')
    else:
        dt = np.mean(np.diff(temp))
        sigma_pixels = sigma / dt
        broadened = gaussian_filter1d(spec_heat, sigma=sigma_pixels, mode='nearest')
    
    return broadened


def run_nlce_triangular(params, fixed_params, exp_temp, work_dir, h_field=None, temp_range=None):
    """
    Run triangular lattice NLCE with the given parameters.
    
    Args:
        params: Parameter array depending on model type:
                - heisenberg/xxz/kitaev: [J1, J2, ...]
                - anisotropic: [Jzz, Jpm, Jpmpm, Jzpm, ...]
        fixed_params: Dictionary of fixed parameters
        exp_temp: Experimental temperature array
        work_dir: Working directory for NLCE calculations
        h_field: Override magnetic field strength
        temp_range: Override temperature range
    
    Returns:
        calc_temp, calc_spec_heat arrays
    """
    model = fixed_params.get("model", "heisenberg")
    
    # Extract parameters based on model type
    if model == 'anisotropic':
        n_model_params = 4
        Jzz, Jpm, Jpmpm, Jzpm = params[:4]
        J1, J2 = 1.0, 0.0  # Not used for anisotropic
    else:
        n_model_params = 2
        J1, J2 = params[:2]
        Jzz, Jpm, Jpmpm, Jzpm = None, None, None, None
    
    h_value = h_field if h_field is not None else fixed_params.get("h", 0.0)
    field_dir = fixed_params["field_dir"]
    
    temp_min = temp_range.get('temp_min', fixed_params["temp_min"]) if temp_range else fixed_params["temp_min"]
    temp_max = temp_range.get('temp_max', fixed_params["temp_max"]) if temp_range else fixed_params["temp_max"]
    
    # Build NLCE command for triangular lattice
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nlce_script = os.path.join(script_dir, '..', 'run', 'nlce_triangular.py')
    
    cmd = [
        'python3',
        nlce_script,
        '--max_order', str(fixed_params["max_order"]),
        '--h', f'{h_value:.12f}',
        '--ed_executable', str(fixed_params["ED_path"]),
        '--field_dir', f'{field_dir[0]:.12f}', f'{field_dir[1]:.12f}', f'{field_dir[2]:.12f}',
        '--temp_min', f'{temp_min:.8f}',
        '--temp_max', f'{temp_max:.8f}',
        '--temp_bins', str(fixed_params["temp_bins"]),
        '--model', model,
        '--thermo',
        '--base_dir', work_dir
    ]
    
    # Add model-specific parameters
    if model == 'anisotropic':
        cmd.extend(['--Jzz', f'{Jzz:.12f}'])
        cmd.extend(['--Jpm', f'{Jpm:.12f}'])
        cmd.extend(['--Jpmpm', f'{Jpmpm:.12f}'])
        cmd.extend(['--Jzpm', f'{Jzpm:.12f}'])
    else:
        cmd.extend(['--J1', f'{J1:.12f}'])
        cmd.extend(['--J2', f'{J2:.12f}'])
    
    if fixed_params.get("skip_cluster_gen", True):
        cmd.append('--skip_cluster_gen')
    
    if not fixed_params.get("skip_ham_prep", False):
        # Clean up old results
        ed_dir = os.path.join(work_dir, f'ed_results_order_{fixed_params["max_order"]}')
        if os.path.exists(ed_dir):
            shutil.rmtree(ed_dir)
        ham_dir = os.path.join(work_dir, f'hamiltonians_order_{fixed_params["max_order"]}')
        if os.path.exists(ham_dir):
            shutil.rmtree(ham_dir)
    else:
        cmd.append('--skip_ham_prep')
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        nlc_dir = os.path.join(work_dir, f'nlc_results_order_{fixed_params["max_order"]}')
        spec_heat_file = os.path.join(nlc_dir, 'nlc_specific_heat.txt')
        
        if not os.path.exists(spec_heat_file):
            logging.error(f"Specific heat file not found: {spec_heat_file}")
            return None, None
        
        calc_data = np.loadtxt(spec_heat_file)
        calc_temp = calc_data[:, 0]
        calc_spec_heat = calc_data[:, 1]
        
        return calc_temp, calc_spec_heat
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running NLCE: {e}")
        logging.error(f"Stderr: {e.stderr.decode('utf-8')}")
        return None, None


def interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp):
    """Interpolate calculated data to match experimental temperature points"""
    if calc_temp is None or calc_spec_heat is None:
        return np.zeros_like(exp_temp)
    
    sort_idx = np.argsort(calc_temp)
    calc_temp = calc_temp[sort_idx]
    calc_spec_heat = calc_spec_heat[sort_idx]
    
    interp_func = interp1d(calc_temp, calc_spec_heat, kind='cubic', 
                          bounds_error=False, fill_value=np.nan)
    
    interp_spec_heat = interp_func(exp_temp)
    interp_spec_heat = np.nan_to_num(interp_spec_heat)
    
    return interp_spec_heat


def calc_chi_squared(params, fixed_params, exp_datasets, work_dir):
    """Calculate chi-squared between experimental and calculated specific heat"""
    total_chi_squared = 0.0
    
    n_datasets = len(exp_datasets)
    fit_broadening = fixed_params.get("fit_broadening", False)
    model = fixed_params.get("model", "heisenberg")
    
    # Determine number of model parameters
    n_model_params = 4 if model == 'anisotropic' else 2
    
    if fit_broadening:
        model_params = params[:n_model_params]
        sigmas = params[n_model_params:n_model_params+n_datasets]
    else:
        model_params = params[:n_model_params]
        sigmas = [0.0] * n_datasets
    
    for i, dataset in enumerate(exp_datasets):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset.get('h', 0.0)
        weight = dataset.get('weight', 1.0)
        
        # Get temperature range for this dataset
        temp_range = {}
        if 'temp_min' in dataset:
            temp_range['temp_min'] = dataset['temp_min']
        if 'temp_max' in dataset:
            temp_range['temp_max'] = dataset['temp_max']
        
        # Run NLCE
        calc_temp, calc_spec_heat = run_nlce_triangular(
            model_params, fixed_params, exp_temp, work_dir, 
            h_field=h_field, temp_range=temp_range if temp_range else None
        )
        
        if calc_temp is None:
            total_chi_squared += 1e10 * weight
            continue
        
        # Interpolate to experimental temperatures
        interp_spec_heat = interpolate_calc_data(calc_temp, calc_spec_heat, exp_temp)
        
        # Apply broadening if fitting
        if fit_broadening and sigmas[i] > 0:
            interp_spec_heat = apply_gaussian_broadening(exp_temp, interp_spec_heat, sigmas[i])
        
        # Filter by temperature range
        temp_mask = np.ones_like(exp_temp, dtype=bool)
        if 'temp_min' in dataset:
            temp_mask &= exp_temp >= dataset['temp_min']
        if 'temp_max' in dataset:
            temp_mask &= exp_temp <= dataset['temp_max']
        
        if np.sum(temp_mask) == 0:
            continue
        
        # Calculate chi-squared
        diff = (exp_spec_heat[temp_mask] - interp_spec_heat[temp_mask])
        chi_sq = np.sum(diff**2) * weight
        total_chi_squared += chi_sq
    
    # Log progress with appropriate parameter names
    model = fixed_params.get("model", "heisenberg")
    if model == 'anisotropic':
        logging.info(f"Parameters: Jzz={params[0]:.4f}, Jpm={params[1]:.4f}, "
                    f"Jpmpm={params[2]:.4f}, Jzpm={params[3]:.4f}, Chi²={total_chi_squared:.4f}")
    else:
        logging.info(f"Parameters: J1={params[0]:.4f}, J2={params[1]:.4f}, Chi²={total_chi_squared:.4f}")
    
    return total_chi_squared


def plot_results(exp_datasets, fixed_params, best_params, work_dir, output_dir):
    """Plot experimental data and best fit"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_datasets)))
    model = fixed_params.get("model", "heisenberg")
    n_model_params = 4 if model == 'anisotropic' else 2
    
    for i, (dataset, color) in enumerate(zip(exp_datasets, colors)):
        exp_temp = dataset['temp']
        exp_spec_heat = dataset['spec_heat']
        h_field = dataset.get('h', 0.0)
        
        # Get temperature range
        temp_range = {}
        if 'temp_min' in dataset:
            temp_range['temp_min'] = dataset['temp_min']
        if 'temp_max' in dataset:
            temp_range['temp_max'] = dataset['temp_max']
        
        # Run NLCE with best parameters
        calc_temp, calc_spec_heat = run_nlce_triangular(
            best_params[:n_model_params], fixed_params, exp_temp, work_dir,
            h_field=h_field, temp_range=temp_range if temp_range else None
        )
        
        # Plot experimental data
        plt.scatter(exp_temp, exp_spec_heat, c=[color], s=30, alpha=0.7, 
                   label=f'Exp (h={h_field})')
        
        # Plot calculated data
        if calc_temp is not None:
            plt.plot(calc_temp, calc_spec_heat, c=color, lw=2, 
                    label=f'NLCE (h={h_field})')
    
    if model == 'anisotropic':
        title_str = (f'Triangular Lattice Fit (Anisotropic): '
                    f'Jzz={best_params[0]:.3f}, Jpm={best_params[1]:.3f}, '
                    f'Jpmpm={best_params[2]:.3f}, Jzpm={best_params[3]:.3f}')
    else:
        title_str = f'Triangular Lattice Fit: J1={best_params[0]:.3f}, J2={best_params[1]:.3f}'
    
    plt.xlabel('Temperature (J₁)')
    plt.ylabel('Specific Heat')
    plt.title(title_str)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_file = os.path.join(output_dir, 'specific_heat_fit.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"Plot saved to {plot_file}")
    plt.close()


def multi_start_optimization(obj_func, initial_params, bounds, n_starts=30, 
                            method='COBYLA', args=(), **kwargs):
    """Perform multi-start optimization"""
    best_result = None
    best_score = np.inf
    all_results = []
    
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    
    # Use Latin Hypercube Sampling
    sampler = qmc.LatinHypercube(d=len(bounds))
    samples = sampler.random(n=n_starts)
    scaled_samples = qmc.scale(samples, lower_bounds, upper_bounds)
    
    logging.info(f"Starting multi-start optimization with {n_starts} starts using {method}")
    
    for i, start_point in enumerate(scaled_samples):
        try:
            result = minimize(
                obj_func,
                start_point,
                method=method,
                bounds=bounds,
                args=args,
                options=kwargs.get('options', {})
            )
            all_results.append(result)
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
            
            logging.info(f"Start {i+1}/{n_starts}: chi²={result.fun:.4f}, params={result.x}")
            
        except Exception as e:
            logging.warning(f"Start {i+1}/{n_starts} failed: {e}")
    
    return best_result


def main():
    parser = argparse.ArgumentParser(description='Fit triangular lattice NLCE to specific heat data')
    
    # Input/output
    parser.add_argument('--exp_data', type=str, help='Path to experimental specific heat data')
    parser.add_argument('--exp_config', type=str, default=None,
                       help='Path to JSON config for multiple datasets')
    parser.add_argument('--output_dir', type=str, default='./fit_results_triangular',
                       help='Output directory')
    parser.add_argument('--work_dir', type=str, default='nlc_fit_triangular_work',
                       help='Working directory')
    parser.add_argument('--ed_executable', type=str, default='./build/ED',
                       help='Path to ED executable')
    
    # Initial guess and bounds
    parser.add_argument('--initial_J1', type=float, default=1.0, help='Initial J1')
    parser.add_argument('--initial_J2', type=float, default=0.0, help='Initial J2')
    parser.add_argument('--J1_min', type=float, default=-3.0, help='Min J1')
    parser.add_argument('--J1_max', type=float, default=3.0, help='Max J1')
    parser.add_argument('--J2_min', type=float, default=-1.0, help='Min J2')
    parser.add_argument('--J2_max', type=float, default=1.0, help='Max J2')
    
    # Anisotropic model initial values and bounds
    parser.add_argument('--initial_Jzz', type=float, default=0.5, help='Initial Jzz')
    parser.add_argument('--initial_Jpm', type=float, default=0.25, help='Initial Jpm')
    parser.add_argument('--initial_Jpmpm', type=float, default=0.05, help='Initial Jpmpm')
    parser.add_argument('--initial_Jzpm', type=float, default=0.1, help='Initial Jzpm')
    parser.add_argument('--Jzz_min', type=float, default=-1.0, help='Min Jzz')
    parser.add_argument('--Jzz_max', type=float, default=1.0, help='Max Jzz')
    parser.add_argument('--Jpm_min', type=float, default=-1.0, help='Min Jpm')
    parser.add_argument('--Jpm_max', type=float, default=1.0, help='Max Jpm')
    parser.add_argument('--Jpmpm_min', type=float, default=-0.5, help='Min Jpmpm')
    parser.add_argument('--Jpmpm_max', type=float, default=0.5, help='Max Jpmpm')
    parser.add_argument('--Jzpm_min', type=float, default=-0.5, help='Min Jzpm')
    parser.add_argument('--Jzpm_max', type=float, default=0.5, help='Max Jzpm')
    
    # NLCE parameters
    parser.add_argument('--max_order', type=int, default=4, help='Maximum NLCE order')
    parser.add_argument('--h', type=float, default=0.0, help='Magnetic field')
    parser.add_argument('--field_dir', type=float, nargs=3, default=[0, 0, 1],
                       help='Field direction')
    parser.add_argument('--temp_bins', type=int, default=100, help='Temperature bins')
    parser.add_argument('--temp_min', type=float, default=0.01, help='Min temperature')
    parser.add_argument('--temp_max', type=float, default=10.0, help='Max temperature')
    parser.add_argument('--model', type=str, default='heisenberg',
                       choices=['heisenberg', 'xxz', 'kitaev', 'anisotropic'],
                       help='Spin model type')
    
    # Skip flags
    parser.add_argument('--skip_cluster_gen', action='store_true')
    parser.add_argument('--skip_ham_prep', action='store_true')
    
    # Optimization
    parser.add_argument('--method', type=str, default='multi_start',
                       choices=['multi_start', 'differential_evolution', 'dual_annealing'])
    parser.add_argument('--n_starts', type=int, default=20, help='Number of random starts')
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations')
    
    # Optional fitting parameters
    parser.add_argument('--fit_broadening', action='store_true')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'nlc_fit_triangular.log')
    setup_logging(log_file)
    
    # Set up working directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    # Load experimental data
    if args.exp_config:
        with open(args.exp_config, 'r') as f:
            config = json.load(f)
        exp_datasets = load_multiple_experimental_data(config['datasets'])
    elif args.exp_data:
        temp, spec_heat = load_experimental_data(args.exp_data)
        exp_datasets = [{
            'temp': temp,
            'spec_heat': spec_heat,
            'h': args.h,
            'field_dir': args.field_dir,
            'weight': 1.0
        }]
    else:
        logging.error("Must provide --exp_data or --exp_config")
        sys.exit(1)
    
    # Fixed parameters
    fixed_params = {
        "max_order": args.max_order,
        "h": args.h,
        "field_dir": args.field_dir,
        "temp_bins": args.temp_bins,
        "temp_min": args.temp_min,
        "temp_max": args.temp_max,
        "model": args.model,
        "ED_path": args.ed_executable,
        "skip_cluster_gen": args.skip_cluster_gen,
        "skip_ham_prep": args.skip_ham_prep,
        "fit_broadening": args.fit_broadening,
        "n_datasets": len(exp_datasets)
    }
    
    # Generate clusters first if needed
    if not args.skip_cluster_gen:
        logging.info("Generating triangular lattice clusters...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cluster_gen_script = os.path.join(script_dir, '..', 'prep', 'generate_triangular_clusters.py')
        cluster_gen_cmd = [
            'python3',
            cluster_gen_script,
            '--max_order', str(args.max_order),
            '--output_dir', os.path.join(args.work_dir, f'clusters_order_{args.max_order}')
        ]
        subprocess.run(cluster_gen_cmd, check=True)
        fixed_params["skip_cluster_gen"] = True
    
    # Set up parameter bounds based on model type
    if args.model == 'anisotropic':
        bounds = [
            (args.Jzz_min, args.Jzz_max),
            (args.Jpm_min, args.Jpm_max),
            (args.Jpmpm_min, args.Jpmpm_max),
            (args.Jzpm_min, args.Jzpm_max)
        ]
        initial_params = [args.initial_Jzz, args.initial_Jpm, args.initial_Jpmpm, args.initial_Jzpm]
        logging.info(f"Fitting anisotropic model: Jzz, Jpm, Jpmpm, Jzpm")
        logging.info(f"Initial: Jzz={args.initial_Jzz}, Jpm={args.initial_Jpm}, "
                    f"Jpmpm={args.initial_Jpmpm}, Jzpm={args.initial_Jzpm}")
    else:
        bounds = [(args.J1_min, args.J1_max), (args.J2_min, args.J2_max)]
        initial_params = [args.initial_J1, args.initial_J2]
        logging.info(f"Fitting {args.model} model: J1, J2")
        logging.info(f"Initial: J1={args.initial_J1}, J2={args.initial_J2}")
    
    if args.fit_broadening:
        for _ in exp_datasets:
            bounds.append((0.0, 1.0))
            initial_params.append(0.1)
    
    # Run optimization
    logging.info("Starting optimization...")
    
    if args.method == 'multi_start':
        result = multi_start_optimization(
            calc_chi_squared,
            initial_params,
            bounds,
            n_starts=args.n_starts,
            method='L-BFGS-B',
            args=(fixed_params, exp_datasets, args.work_dir),
            options={'maxiter': args.max_iter}
        )
    elif args.method == 'differential_evolution':
        result = differential_evolution(
            calc_chi_squared,
            bounds,
            args=(fixed_params, exp_datasets, args.work_dir),
            maxiter=args.max_iter,
            seed=42,
            disp=True
        )
    elif args.method == 'dual_annealing':
        result = dual_annealing(
            calc_chi_squared,
            bounds,
            args=(fixed_params, exp_datasets, args.work_dir),
            maxiter=args.max_iter,
            seed=42
        )
    
    if result is None:
        logging.error("Optimization failed!")
        sys.exit(1)
    
    best_params = result.x
    best_chi_sq = result.fun
    
    logging.info("="*80)
    logging.info("Optimization completed!")
    
    if args.model == 'anisotropic':
        logging.info(f"Best Jzz: {best_params[0]:.6f}")
        logging.info(f"Best Jpm: {best_params[1]:.6f}")
        logging.info(f"Best Jpmpm: {best_params[2]:.6f}")
        logging.info(f"Best Jzpm: {best_params[3]:.6f}")
        results_dict = {
            'best_params': {
                'Jzz': float(best_params[0]),
                'Jpm': float(best_params[1]),
                'Jpmpm': float(best_params[2]),
                'Jzpm': float(best_params[3])
            },
            'chi_squared': float(best_chi_sq),
            'fixed_params': fixed_params
        }
    else:
        logging.info(f"Best J1: {best_params[0]:.6f}")
        logging.info(f"Best J2: {best_params[1]:.6f}")
        results_dict = {
            'best_params': {
                'J1': float(best_params[0]),
                'J2': float(best_params[1])
            },
            'chi_squared': float(best_chi_sq),
            'fixed_params': fixed_params
        }
    
    logging.info(f"Best chi-squared: {best_chi_sq:.6f}")
    logging.info("="*80)
    
    # Save results
    with open(os.path.join(args.output_dir, 'fit_results.json'), 'w') as f:
        json.dump(results_dict, f, indent=2, cls=NumpyJSONEncoder)
    
    # Plot results
    plot_results(exp_datasets, fixed_params, best_params, args.work_dir, args.output_dir)
    
    logging.info("Fitting completed successfully!")


if __name__ == "__main__":
    main()
