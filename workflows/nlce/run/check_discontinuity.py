#!/usr/bin/env python3
"""
Check NLC output for discontinuities in resummed results.
"""

import numpy as np
import sys

def check_discontinuity(filename, property_name):
    """Load data and check for discontinuities."""
    
    # Load data (skip comment lines)
    data = np.loadtxt(filename, comments='#')
    temps = data[:, 0]
    values = data[:, 1]  # Column 1 is resummed value
    
    # Check for discontinuities
    diffs = np.diff(values)
    diff_of_diffs = np.diff(diffs)
    
    # Statistics
    mean_diff = np.mean(np.abs(diffs))
    max_diff = np.max(np.abs(diffs))
    max_jump = np.max(np.abs(diff_of_diffs))
    
    # Expected scale for smooth function
    expected_scale = ((temps[-1] - temps[0]) / len(temps))**2
    jump_threshold = 100 * expected_scale
    
    print(f"\n{'='*80}")
    print(f"Checking: {property_name}")
    print(f"File: {filename}")
    print('='*80)
    print(f"  Data points: {len(values)}")
    print(f"  T range: [{temps[0]:.4f}, {temps[-1]:.4f}]")
    print(f"  Value range: [{np.min(values):.6f}, {np.max(values):.6f}]")
    print(f"  Mean |Δf|: {mean_diff:.6e}")
    print(f"  Max |Δf|: {max_diff:.6e}")
    print(f"  Max |Δ²f|: {max_jump:.6e}")
    print(f"  Jump threshold: {jump_threshold:.6e}")
    
    # Check for large jumps
    if max_jump > jump_threshold:
        print(f"  ✗ DISCONTINUITY DETECTED!")
        jump_idx = np.argmax(np.abs(diff_of_diffs))
        T_jump = temps[jump_idx+1]
        print(f"    Jump at T ≈ {T_jump:.4f} (index {jump_idx+1})")
        print(f"    Values around jump:")
        for i in range(max(0, jump_idx-2), min(len(values), jump_idx+4)):
            marker = " ← JUMP" if i == jump_idx+1 else ""
            print(f"      T={temps[i]:.4f}: {values[i]:.8f}{marker}")
        return False
    else:
        print(f"  ✓ SMOOTH: No discontinuities detected")
        return True

if __name__ == '__main__':
    base_dir = "./nlce_ftlm_convergence/order_6/nlc_results_order_6"
    
    properties = [
        ('nlc_energy.txt', 'Energy'),
        ('nlc_specific_heat.txt', 'Specific Heat'),
        ('nlc_entropy.txt', 'Entropy'),
        ('nlc_free_energy.txt', 'Free Energy')
    ]
    
    all_smooth = True
    for filename, prop_name in properties:
        try:
            is_smooth = check_discontinuity(f"{base_dir}/{filename}", prop_name)
            all_smooth = all_smooth and is_smooth
        except Exception as e:
            print(f"\n✗ Error checking {prop_name}: {e}")
            all_smooth = False
    
    print(f"\n{'='*80}")
    if all_smooth:
        print("✓✓✓ ALL PROPERTIES ARE SMOOTH ✓✓✓")
    else:
        print("✗✗✗ DISCONTINUITIES DETECTED ✗✗✗")
    print('='*80)
    
    sys.exit(0 if all_smooth else 1)
