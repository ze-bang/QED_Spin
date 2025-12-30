#!/usr/bin/env python3
"""
Test continuity of NLC resummation - verify no discontinuities across temperatures.
"""

import numpy as np
import sys
import os

# Add workflows to path
sys.path.insert(0, os.path.dirname(__file__))
from NLC_sum_ftlm import NLCExpansionFTLM

def test_continuity():
    """Test that resummation produces smooth curves without jumps."""
    
    # Create dummy NLC expansion object to access resummation methods
    nlc = NLCExpansionFTLM(
        cluster_dir="dummy",
        ftlm_dir="dummy", 
        temp_min=0.1,
        temp_max=10.0,
        num_temps=100
    )
    
    print("="*80)
    print("CONTINUITY TEST: Verify no discontinuities across temperatures")
    print("="*80)
    
    # Test case: Alternating series with temperature dependence
    # This mimics NLC where different orders have different T-dependence
    # but should produce SMOOTH output after resummation
    n_orders = 6
    n_temps = len(nlc.temp_values)
    
    # Create partial sums with realistic T-dependence
    # Each order contributes: (-1)^n * T^n / (1 + T)^{n+1}
    partial_sums = []
    for n in range(n_orders):
        # Alternating series with decay
        sign = (-1)**n
        term = sign * nlc.temp_values**n / (1 + nlc.temp_values)**(n+1)
        
        if n == 0:
            partial_sums.append(term.copy())
        else:
            partial_sums.append(partial_sums[-1] + term)
    
    # Test each method
    methods = ['euler', 'wynn', 'auto']
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Method: {method.upper()}")
        print('='*80)
        
        result = nlc.apply_resummation(partial_sums, method=method, l_euler=3)
        values = result['value']
        
        # Check for discontinuities: compute finite differences
        diffs = np.diff(values)
        diff_of_diffs = np.diff(diffs)
        
        # Statistics
        mean_diff = np.mean(np.abs(diffs))
        max_diff = np.max(np.abs(diffs))
        max_jump = np.max(np.abs(diff_of_diffs))
        
        # A discontinuity shows up as a large second derivative
        # For a smooth function, |Δ²f| should be O(Δx²) ≈ (T_max - T_min)² / N²
        expected_scale = ((nlc.temp_values[-1] - nlc.temp_values[0]) / n_temps)**2
        jump_threshold = 100 * expected_scale  # Allow 100x the expected scale
        
        print(f"  Values range: [{np.min(values):.6f}, {np.max(values):.6f}]")
        print(f"  Mean |Δf|: {mean_diff:.6e}")
        print(f"  Max |Δf|: {max_diff:.6e}")
        print(f"  Max |Δ²f|: {max_jump:.6e}")
        print(f"  Jump threshold: {jump_threshold:.6e}")
        
        # Check for jumps
        if max_jump > jump_threshold:
            print(f"  ✗ DISCONTINUITY DETECTED!")
            print(f"    Maximum jump |Δ²f| = {max_jump:.2e} exceeds threshold {jump_threshold:.2e}")
            
            # Find where the jump occurs
            jump_idx = np.argmax(np.abs(diff_of_diffs))
            T_jump = nlc.temp_values[jump_idx+1]
            print(f"    Jump occurs at T ≈ {T_jump:.3f}")
            print(f"    Values: f({T_jump-0.1:.2f})={values[jump_idx]:.6f}, f({T_jump:.2f})={values[jump_idx+1]:.6f}, f({T_jump+0.1:.2f})={values[jump_idx+2]:.6f}")
            
            return False
        else:
            print(f"  ✓ SMOOTH: No discontinuities detected")
    
    print(f"\n{'='*80}")
    print("✓✓✓ ALL METHODS PRODUCE CONTINUOUS OUTPUT ✓✓✓")
    print('='*80)
    return True

if __name__ == '__main__':
    success = test_continuity()
    sys.exit(0 if success else 1)
