#!/usr/bin/env python3
"""
Test GPU vs CPU FTLM thermal spectral function implementation.

This script runs the FTLM thermal spectral calculation using both CPU and GPU
and compares the results to verify consistency.
"""

import subprocess
import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import argparse

# Paths
BASE_DIR = Path(__file__).parent.parent
BUILD_DIR = BASE_DIR / "build"
TEST_DIR = BASE_DIR / "test_4_sites"

def run_ftlm_spectral(use_gpu: bool, output_subdir: str, num_samples: int = 20, 
                       krylov_dim: int = 50, temperatures: list = None):
    """Run FTLM thermal spectral calculation."""
    
    executable = BUILD_DIR / "TPQ_DSSF"
    if not executable.exists():
        print(f"Error: {executable} not found. Please build first.")
        sys.exit(1)
    
    output_dir = TEST_DIR / output_subdir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        str(executable),
        str(TEST_DIR),
        str(krylov_dim),  # Krylov dimension
        "2,2",  # Operator pair (Sz-Sz)
        "ftlm_thermal",  # Method
        "sum",  # observable_type
        "ladder",  # momentum_pattern
        "--omega-min=0.0",
        "--omega-max=5.0",
        "--num-omega=100",
        "--broadening=0.1",
        f"--num-samples={num_samples}",
        "--random-seed=42",  # Fixed seed for reproducibility
    ]
    
    if temperatures:
        cmd.append(f"--T-min={min(temperatures)}")
        cmd.append(f"--T-max={max(temperatures)}")
        cmd.append(f"--T-steps={len(temperatures)}")
    else:
        cmd.append("--T-min=0.5")
        cmd.append("--T-max=2.0")
        cmd.append("--T-steps=3")
    
    if use_gpu:
        cmd.append("--use-gpu")
    else:
        cmd.append("--no-gpu")
    
    # Set output directory
    os.environ['OUTPUT_DIR'] = str(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Running FTLM thermal spectral ({'GPU' if use_gpu else 'CPU'})")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=str(TEST_DIR),
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("\n--- STDOUT ---")
        print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        
        if result.returncode != 0:
            print("\n--- STDERR ---")
            print(result.stderr)
            print(f"Warning: Command exited with code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("Error: Command timed out")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None
    
    # Find HDF5 output file
    h5_files = list(output_dir.glob("**/*.h5")) + list(TEST_DIR.glob("**/*dssf*.h5"))
    if h5_files:
        return h5_files[0]
    
    # Check test directory for results
    dssf_file = TEST_DIR / "dssf_results.h5"
    if dssf_file.exists():
        return dssf_file
    
    print(f"Warning: No HDF5 output file found in {output_dir}")
    return None


def load_ftlm_results(h5_path):
    """Load FTLM spectral results from HDF5 file."""
    results = {}
    
    if h5_path is None or not Path(h5_path).exists():
        print(f"Error: HDF5 file not found: {h5_path}")
        return results
    
    print(f"\nLoading results from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"HDF5 structure: {list(f.keys())}")
        
        def visit_items(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name} shape={obj.shape}")
        
        f.visititems(visit_items)
        
        # Try to find spectral function data
        for key in f.keys():
            if 'spectral' in key.lower() or 'dssf' in key.lower() or 'ftlm' in key.lower():
                group = f[key]
                if isinstance(group, h5py.Group):
                    for subkey in group.keys():
                        if 'frequencies' in subkey.lower() or 'omega' in subkey.lower():
                            results['frequencies'] = group[subkey][:]
                        elif 'spectral' in subkey.lower() or 's_real' in subkey.lower():
                            results['spectral'] = group[subkey][:]
                            
            # Check for temperature-dependent data
            if 'T=' in key or 'temperature' in key.lower():
                temp_data = {}
                group = f[key]
                if isinstance(group, h5py.Group):
                    for subkey in group.keys():
                        temp_data[subkey] = group[subkey][:]
                results[key] = temp_data
                
    return results


def compare_results(cpu_results, gpu_results, tolerance=0.1):
    """Compare CPU and GPU results."""
    
    print(f"\n{'='*60}")
    print("Comparing CPU vs GPU Results")
    print(f"{'='*60}")
    
    if not cpu_results or not gpu_results:
        print("Error: Missing results for comparison")
        return False
    
    print(f"CPU results keys: {list(cpu_results.keys())}")
    print(f"GPU results keys: {list(gpu_results.keys())}")
    
    all_passed = True
    
    # Compare frequencies
    if 'frequencies' in cpu_results and 'frequencies' in gpu_results:
        freq_diff = np.max(np.abs(cpu_results['frequencies'] - gpu_results['frequencies']))
        print(f"\nFrequency grid difference: {freq_diff:.2e}")
        if freq_diff > 1e-10:
            print("  WARNING: Frequency grids differ!")
            all_passed = False
    
    # Compare spectral functions
    if 'spectral' in cpu_results and 'spectral' in gpu_results:
        cpu_spec = cpu_results['spectral']
        gpu_spec = gpu_results['spectral']
        
        # Normalize for comparison
        cpu_max = np.max(np.abs(cpu_spec)) if np.max(np.abs(cpu_spec)) > 0 else 1.0
        gpu_max = np.max(np.abs(gpu_spec)) if np.max(np.abs(gpu_spec)) > 0 else 1.0
        
        rel_diff = np.max(np.abs(cpu_spec/cpu_max - gpu_spec/gpu_max))
        mean_diff = np.mean(np.abs(cpu_spec - gpu_spec))
        
        print(f"\nSpectral function comparison:")
        print(f"  Max relative difference: {rel_diff:.4f}")
        print(f"  Mean absolute difference: {mean_diff:.6f}")
        print(f"  CPU max value: {cpu_max:.6f}")
        print(f"  GPU max value: {gpu_max:.6f}")
        
        if rel_diff > tolerance:
            print(f"  WARNING: Relative difference {rel_diff:.4f} > tolerance {tolerance}")
            all_passed = False
        else:
            print(f"  PASS: Results within tolerance")
    
    return all_passed


def plot_comparison(cpu_h5, gpu_h5, output_path):
    """Plot comparison of CPU and GPU results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Load and plot data
    for h5_path, label, color in [(cpu_h5, 'CPU', 'blue'), (gpu_h5, 'GPU', 'red')]:
        if h5_path is None or not Path(h5_path).exists():
            continue
            
        with h5py.File(h5_path, 'r') as f:
            # Try to find spectral data
            frequencies = None
            spectral = None
            
            def find_data(name, obj):
                nonlocal frequencies, spectral
                if isinstance(obj, h5py.Dataset):
                    if 'freq' in name.lower() or 'omega' in name.lower():
                        frequencies = obj[:]
                    elif 'spectral' in name.lower() and 'error' not in name.lower():
                        if spectral is None or 'real' in name.lower():
                            spectral = obj[:]
            
            f.visititems(find_data)
            
            if frequencies is not None and spectral is not None:
                # Handle multi-dimensional spectral data
                if spectral.ndim == 1:
                    axes[0, 0].plot(frequencies, spectral, label=label, color=color, alpha=0.7)
                elif spectral.ndim == 2:
                    for i in range(min(spectral.shape[0], 3)):
                        axes[0, 0].plot(frequencies, spectral[i], 
                                       label=f'{label} T{i}', color=color, 
                                       alpha=0.7, linestyle=['-', '--', ':'][i])
    
    axes[0, 0].set_xlabel('ω')
    axes[0, 0].set_ylabel('S(ω)')
    axes[0, 0].set_title('FTLM Spectral Function Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot difference if both exist
    if cpu_h5 and gpu_h5 and Path(cpu_h5).exists() and Path(gpu_h5).exists():
        with h5py.File(cpu_h5, 'r') as fc, h5py.File(gpu_h5, 'r') as fg:
            cpu_freq, cpu_spec = None, None
            gpu_freq, gpu_spec = None, None
            
            def find_cpu(name, obj):
                nonlocal cpu_freq, cpu_spec
                if isinstance(obj, h5py.Dataset):
                    if 'freq' in name.lower():
                        cpu_freq = obj[:]
                    elif 'spectral' in name.lower() and 'error' not in name.lower():
                        cpu_spec = obj[:]
            
            def find_gpu(name, obj):
                nonlocal gpu_freq, gpu_spec
                if isinstance(obj, h5py.Dataset):
                    if 'freq' in name.lower():
                        gpu_freq = obj[:]
                    elif 'spectral' in name.lower() and 'error' not in name.lower():
                        gpu_spec = obj[:]
            
            fc.visititems(find_cpu)
            fg.visititems(find_gpu)
            
            if cpu_spec is not None and gpu_spec is not None:
                if cpu_spec.shape == gpu_spec.shape:
                    diff = cpu_spec - gpu_spec
                    if diff.ndim == 1:
                        axes[0, 1].plot(cpu_freq, diff, 'g-')
                    else:
                        for i in range(min(diff.shape[0], 3)):
                            axes[0, 1].plot(cpu_freq, diff[i], 
                                           label=f'T{i}', alpha=0.7)
                    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    axes[0, 1].set_xlabel('ω')
                    axes[0, 1].set_ylabel('CPU - GPU')
                    axes[0, 1].set_title('Difference')
                    axes[0, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test GPU vs CPU FTLM implementation')
    parser.add_argument('--samples', type=int, default=10, help='Number of random samples')
    parser.add_argument('--krylov', type=int, default=30, help='Krylov dimension')
    parser.add_argument('--plot', action='store_true', help='Generate comparison plot')
    parser.add_argument('--tolerance', type=float, default=0.15, 
                        help='Tolerance for relative difference')
    args = parser.parse_args()
    
    print("="*60)
    print("GPU vs CPU FTLM Thermal Spectral Function Test")
    print("="*60)
    print(f"Test directory: {TEST_DIR}")
    print(f"Samples: {args.samples}")
    print(f"Krylov dimension: {args.krylov}")
    
    # Check if test directory exists
    if not TEST_DIR.exists():
        print(f"Error: Test directory {TEST_DIR} not found")
        sys.exit(1)
    
    # Run CPU version
    print("\n" + "="*60)
    print("STEP 1: Running CPU implementation")
    print("="*60)
    cpu_h5 = run_ftlm_spectral(
        use_gpu=False, 
        output_subdir="ftlm_cpu_test",
        num_samples=args.samples,
        krylov_dim=args.krylov,
        temperatures=[0.5, 1.0, 2.0]
    )
    
    # Run GPU version  
    print("\n" + "="*60)
    print("STEP 2: Running GPU implementation")
    print("="*60)
    gpu_h5 = run_ftlm_spectral(
        use_gpu=True,
        output_subdir="ftlm_gpu_test", 
        num_samples=args.samples,
        krylov_dim=args.krylov,
        temperatures=[0.5, 1.0, 2.0]
    )
    
    # Load and compare results
    print("\n" + "="*60)
    print("STEP 3: Comparing results")
    print("="*60)
    
    cpu_results = load_ftlm_results(cpu_h5)
    gpu_results = load_ftlm_results(gpu_h5)
    
    passed = compare_results(cpu_results, gpu_results, tolerance=args.tolerance)
    
    # Generate plot if requested
    if args.plot:
        plot_path = TEST_DIR / "gpu_vs_cpu_comparison.png"
        plot_comparison(cpu_h5, gpu_h5, plot_path)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"CPU output: {cpu_h5}")
    print(f"GPU output: {gpu_h5}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
