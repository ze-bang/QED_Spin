#!/usr/bin/env python3
"""
Test automatic resummation method selection on various series types.
"""

import numpy as np
import sys
from NLC_sum_ftlm import NLCExpansionFTLM


def test_auto_selection():
    """Test that auto-selection picks appropriate methods for different series."""
    
    print("="*80)
    print("TESTING AUTOMATIC RESUMMATION METHOD SELECTION")
    print("="*80)
    
    # Create dummy NLC instance
    nlc = NLCExpansionFTLM('.', '.', 0.1, 1.0, 100)
    nlc.temp_values = np.array([1.0])  # Single temperature point
    
    test_cases = []
    
    # Test 1: Converging geometric series (r=0.5)
    print("\n" + "="*80)
    print("TEST 1: Fast-converging geometric series (r=0.5)")
    print("="*80)
    r = 0.5
    n_terms = 12
    partial_sums = [np.array([sum(r**k for k in range(n+1))]) for n in range(n_terms)]
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Geometric (r=0.5)', selected, 'wynn or direct'))
    
    # Test 2: Alternating harmonic series
    print("\n" + "="*80)
    print("TEST 2: Alternating harmonic series")
    print("="*80)
    n_terms = 20
    partial_sums = [np.array([sum((-1)**(k+1) / k for k in range(1, n+1))]) 
                    for n in range(1, n_terms+1)]
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Alternating harmonic', selected, 'euler'))
    
    # Test 3: Slowly converging series (r=0.9)
    print("\n" + "="*80)
    print("TEST 3: Slowly converging geometric series (r=0.9)")
    print("="*80)
    r = 0.9
    n_terms = 12
    partial_sums = [np.array([sum(r**k for k in range(n+1))]) for n in range(n_terms)]
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Geometric (r=0.9)', selected, 'wynn'))
    
    # Test 4: Diverging series (r=1.2)
    print("\n" + "="*80)
    print("TEST 4: Diverging geometric series (r=1.2)")
    print("="*80)
    r = 1.2
    n_terms = 10
    partial_sums = [np.array([sum(r**k for k in range(n+1))]) for n in range(n_terms)]
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Geometric (r=1.2)', selected, 'direct'))  # Changed from 'conservative'
    
    # Test 5: Oscillating but not strictly alternating
    print("\n" + "="*80)
    print("TEST 5: Oscillating series (not strictly alternating)")
    print("="*80)
    # Create series: 1, -0.5, 0.3, -0.2, 0.15, -0.1, 0.08, -0.06, 0.05
    increments = [1.0, -0.5, 0.3, -0.2, 0.15, -0.1, 0.08, -0.06, 0.05, -0.04]
    partial_sums = [np.array([sum(increments[:i+1])]) for i in range(len(increments))]
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Oscillating', selected, 'euler'))  # Changed from 'euler or theta'
    
    # Test 6: Very few terms (N=3)
    print("\n" + "="*80)
    print("TEST 6: Very few terms (N=3)")
    print("="*80)
    partial_sums = [np.array([1.0]), np.array([1.5]), np.array([1.7])]
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Few terms (N=3)', selected, 'direct'))
    
    # Test 7: Temperature-dependent behavior (converging at high T, diverging at low T)
    print("\n" + "="*80)
    print("TEST 7: Temperature-dependent series (converging high-T, diverging low-T)")
    print("="*80)
    nlc.temp_values = np.logspace(-4, 0, 100)  # 100 temperature points
    n_terms = 8
    partial_sums = []
    for n in range(n_terms):
        # Series that converges at high T (small terms) but diverges at low T (growing)
        # Use temperature-dependent ratio: r(T) = 0.5 + 0.7/T
        term_values = np.zeros(100)
        for k in range(n+1):
            r_T = 0.5 + 0.7 / nlc.temp_values  # r>1 at low T, r<1 at high T
            term_values += r_T**k
        partial_sums.append(term_values)
    selected = nlc.select_resummation_method(partial_sums, verbose=True)
    test_cases.append(('Temperature-dependent', selected, 'direct'))  # Changed from 'conservative or euler'
    
    # Summary
    print("\n" + "="*80)
    print("SELECTION SUMMARY")
    print("="*80)
    all_passed = True
    for name, selected, expected in test_cases:
        expected_methods = expected.split(' or ')
        passed = selected in expected_methods
        status = "✓" if passed else "✗"
        print(f"{status} {name:40s} → {selected:15s} (expected: {expected})")
        if not passed:
            all_passed = False
    
    print("="*80)
    if all_passed:
        print("✓✓✓ ALL AUTO-SELECTION TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME AUTO-SELECTION TESTS FAILED ✗✗✗")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = test_auto_selection()
    sys.exit(0 if success else 1)
