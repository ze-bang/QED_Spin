#!/usr/bin/env python3
"""
Fix FTLM Entropy Bug

This script applies the entropy fix to existing FTLM H5 files by:
1. Reading the averaged energy and specific heat (which are correctly averaged)
2. Recomputing entropy by integrating specific heat: S(T) = ∫₀^T (C_v/T') dT'
3. Recomputing free energy from F = E - TS
4. Saving the corrected values back to the H5 file

The bug was that FTLM entropy and free energy were being directly averaged
across samples, but each sample used a different reference energy E_min.
This led to negative entropies and thermodynamically inconsistent results.
"""

import os
import sys
import glob
import argparse
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Error: h5py is required. Install with: pip install h5py")
    sys.exit(1)


def fix_entropy_in_h5(h5_path, verbose=True):
    """
    Fix the entropy and free energy in an FTLM H5 file.
    
    Args:
        h5_path: Path to the H5 file
        verbose: Print diagnostic information
        
    Returns:
        dict with before/after statistics
    """
    stats = {'path': h5_path, 'fixed': False}
    
    try:
        with h5py.File(h5_path, 'r+') as h5:
            # Check for FTLM averaged data
            if '/ftlm/averaged' not in h5:
                stats['error'] = 'No /ftlm/averaged group found'
                return stats
            
            ftlm = h5['/ftlm/averaged']
            
            # Read existing data
            temps = ftlm['temperatures'][:]
            energy = ftlm['energy'][:]
            spec_heat = ftlm['specific_heat'][:]
            old_entropy = ftlm['entropy'][:]
            old_free_energy = ftlm['free_energy'][:]
            
            n_temps = len(temps)
            
            # Compute fixed entropy by integrating specific heat
            # S(T) = ∫₀^T (C_v/T') dT'
            fixed_entropy = np.zeros(n_temps)
            fixed_entropy[0] = 0.0  # S(T→0) = 0 for non-degenerate ground state
            
            for t in range(1, n_temps):
                T_prev = temps[t-1]
                T_curr = temps[t]
                log_T_diff = np.log(T_curr) - np.log(T_prev)
                Cv_avg = (spec_heat[t-1] + spec_heat[t]) / 2.0
                fixed_entropy[t] = fixed_entropy[t-1] + Cv_avg * log_T_diff
            
            # Compute fixed free energy from F = E - TS
            fixed_free_energy = energy - temps * fixed_entropy
            
            # Store stats for reporting
            stats['old_entropy_min'] = float(old_entropy.min())
            stats['old_entropy_max'] = float(old_entropy.max())
            stats['new_entropy_min'] = float(fixed_entropy.min())
            stats['new_entropy_max'] = float(fixed_entropy.max())
            stats['old_free_energy_low_T'] = float(old_free_energy[0])
            stats['new_free_energy_low_T'] = float(fixed_free_energy[0])
            stats['energy_low_T'] = float(energy[0])
            
            # Verify thermodynamic consistency
            stats['energy_minus_free_energy_low_T'] = float(energy[0] - fixed_free_energy[0])
            
            # Overwrite entropy and free_energy datasets
            ftlm['entropy'][...] = fixed_entropy
            ftlm['free_energy'][...] = fixed_free_energy
            
            # Also zero out entropy and free_energy errors since they're now derived
            if 'entropy_error' in ftlm:
                ftlm['entropy_error'][...] = np.zeros(n_temps)
            if 'free_energy_error' in ftlm:
                ftlm['free_energy_error'][...] = np.zeros(n_temps)
            
            stats['fixed'] = True
            
            if verbose:
                print(f"Fixed {h5_path}:")
                print(f"  Old entropy range: [{stats['old_entropy_min']:.4f}, {stats['old_entropy_max']:.4f}]")
                print(f"  New entropy range: [{stats['new_entropy_min']:.4f}, {stats['new_entropy_max']:.4f}]")
                print(f"  Low-T check: E={energy[0]:.4f}, F_new={fixed_free_energy[0]:.4f}, S_new={fixed_entropy[0]:.4f}")
            
    except Exception as e:
        stats['error'] = str(e)
        if verbose:
            print(f"Error processing {h5_path}: {e}")
    
    return stats


def fix_all_ftlm_files(base_dir, pattern="**/ed_results.h5", verbose=True):
    """
    Find and fix all FTLM H5 files in a directory tree.
    
    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for H5 files
        verbose: Print diagnostic information
        
    Returns:
        list of stats dicts
    """
    h5_files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
    
    if verbose:
        print(f"Found {len(h5_files)} H5 files to process")
    
    all_stats = []
    for h5_path in h5_files:
        stats = fix_entropy_in_h5(h5_path, verbose=verbose)
        all_stats.append(stats)
    
    # Summary
    fixed_count = sum(1 for s in all_stats if s.get('fixed', False))
    error_count = sum(1 for s in all_stats if 'error' in s)
    
    if verbose:
        print(f"\nSummary: Fixed {fixed_count} files, {error_count} errors, {len(h5_files) - fixed_count - error_count} skipped")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(description='Fix FTLM entropy bug in H5 files')
    parser.add_argument('path', help='Path to H5 file or directory')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Recursively search for H5 files')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    if os.path.isfile(args.path):
        fix_entropy_in_h5(args.path, verbose=not args.quiet)
    elif os.path.isdir(args.path):
        if args.recursive:
            fix_all_ftlm_files(args.path, verbose=not args.quiet)
        else:
            h5_files = glob.glob(os.path.join(args.path, "*.h5"))
            for h5_path in h5_files:
                fix_entropy_in_h5(h5_path, verbose=not args.quiet)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    main()
