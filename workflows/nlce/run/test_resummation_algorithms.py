#!/usr/bin/env python3
"""
Validation tests for resummation algorithms.

Tests the Euler, Wynn epsilon, and Brezinski theta accelerators
on sequences with known limits (geometric series, harmonic, etc.)
to verify correct implementation.
"""

import numpy as np
import sys


def test_geometric_series(verbose=True):
    """
    Test on geometric series: S = sum_{n=0}^{infty} r^n = 1/(1-r)
    For r = 0.5, exact sum = 2.0
    """
    r = 0.5
    exact_sum = 1.0 / (1.0 - r)
    
    # Generate partial sums - need enough for Wynn to work
    n_terms = 12
    partial_sums = [sum(r**k for k in range(n+1)) for n in range(n_terms)]
    
    if verbose:
        print("="*80)
        print("TEST: Geometric Series (r=0.5)")
        print("="*80)
        print(f"Exact sum: {exact_sum:.10f}")
        print(f"Direct (N={n_terms}): {partial_sums[-1]:.10f}, error: {abs(partial_sums[-1] - exact_sum):.2e}")
        print(f"\nLast 5 partial sums: {[f'{s:.6f}' for s in partial_sums[-5:]]}")
        print(f"Last increment: {partial_sums[-1] - partial_sums[-2]:.6e}")
    
    # Test Wynn epsilon
    try:
        from NLC_sum_ftlm import NLCExpansionFTLM
        
        # Create dummy instance to access methods
        nlc = NLCExpansionFTLM('.', '.', 0.1, 1.0, 100)
        nlc.temp_values = np.array([1.0])  # Single temperature point
        
        # Convert to array with shape (n_terms, 1) for temperature dimension
        ps_array = [np.array([s]) for s in partial_sums]
        
        wynn_result, wynn_err = nlc.wynn_epsilon(ps_array)
        wynn_val = wynn_result[0]
        
        # Also get all evens for diagnostic
        all_wynn_evens = nlc.wynn_epsilon(ps_array, return_all_evens=True)
        
        euler_result, euler_err = nlc.euler_resummation(ps_array, l=3)
        euler_val = euler_result[0]
        
        theta_result, theta_err = nlc.brezinski_theta(ps_array)
        theta_val = theta_result[0]
        
        if verbose:
            print(f"\nWynn ε: {wynn_val:.10f}, error: {abs(wynn_val - exact_sum):.2e}")
            if len(all_wynn_evens) > 0:
                print(f"  All Wynn evens: {[f'{e[0]:.6f}' for e in all_wynn_evens]}")
            print(f"Euler:  {euler_val:.10f}, error: {abs(euler_val - exact_sum):.2e}")
            print(f"Theta:  {theta_val:.10f}, error: {abs(theta_val - exact_sum):.2e}")
        
        # Check if Wynn is better than direct (should be for geometric series)
        wynn_better = abs(wynn_val - exact_sum) < abs(partial_sums[-1] - exact_sum)
        
        if verbose:
            print(f"\n✓ Wynn acceleration successful: {wynn_better}")
        
        return {
            'wynn_error': abs(wynn_val - exact_sum),
            'euler_error': abs(euler_val - exact_sum),
            'theta_error': abs(theta_val - exact_sum),
            'direct_error': abs(partial_sums[-1] - exact_sum),
            'wynn_better': wynn_better
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_alternating_harmonic(verbose=True):
    """
    Test on alternating harmonic series: S = sum_{n=1}^{infty} (-1)^{n+1}/n = ln(2)
    """
    exact_sum = np.log(2)
    
    # Generate partial sums
    n_terms = 20
    partial_sums = [sum((-1)**(k+1) / k for k in range(1, n+1)) for n in range(1, n_terms+1)]
    
    if verbose:
        print("\n" + "="*80)
        print("TEST: Alternating Harmonic Series")
        print("="*80)
        print(f"Exact sum: {exact_sum:.10f}")
        print(f"Direct (N={n_terms}): {partial_sums[-1]:.10f}, error: {abs(partial_sums[-1] - exact_sum):.2e}")
    
    try:
        from NLC_sum_ftlm import NLCExpansionFTLM
        
        nlc = NLCExpansionFTLM('.', '.', 0.1, 1.0, 100)
        nlc.temp_values = np.array([1.0])
        
        ps_array = [np.array([s]) for s in partial_sums]
        
        wynn_result, wynn_err = nlc.wynn_epsilon(ps_array)
        wynn_val = wynn_result[0]
        
        euler_result, euler_err = nlc.euler_resummation(ps_array, l=5)
        euler_val = euler_result[0]
        
        theta_result, theta_err = nlc.brezinski_theta(ps_array)
        theta_val = theta_result[0]
        
        if verbose:
            print(f"\nWynn ε: {wynn_val:.10f}, error: {abs(wynn_val - exact_sum):.2e}")
            print(f"Euler:  {euler_val:.10f}, error: {abs(euler_val - exact_sum):.2e}")
            print(f"Theta:  {theta_val:.10f}, error: {abs(theta_val - exact_sum):.2e}")
        
        # For alternating series, Euler should be best
        euler_better = abs(euler_val - exact_sum) < abs(partial_sums[-1] - exact_sum)
        
        if verbose:
            print(f"\n✓ Euler acceleration successful: {euler_better}")
            print("  (Euler is designed for alternating series)")
        
        return {
            'wynn_error': abs(wynn_val - exact_sum),
            'euler_error': abs(euler_val - exact_sum),
            'theta_error': abs(theta_val - exact_sum),
            'direct_error': abs(partial_sums[-1] - exact_sum),
            'euler_better': euler_better
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_exponential_series(verbose=True):
    """
    Test on exponential series: S = sum_{n=0}^{infty} x^n/n! = e^x
    For x = 1, exact sum = e
    """
    x = 1.0
    exact_sum = np.exp(x)
    
    # Generate partial sums
    n_terms = 15
    partial_sums = []
    import math
    for n in range(n_terms):
        s = sum(x**k / math.factorial(k) for k in range(n+1))
        partial_sums.append(s)
    
    if verbose:
        print("\n" + "="*80)
        print("TEST: Exponential Series (e^1)")
        print("="*80)
        print(f"Exact sum: {exact_sum:.10f}")
        print(f"Direct (N={n_terms}): {partial_sums[-1]:.10f}, error: {abs(partial_sums[-1] - exact_sum):.2e}")
    
    try:
        from NLC_sum_ftlm import NLCExpansionFTLM
        
        nlc = NLCExpansionFTLM('.', '.', 0.1, 1.0, 100)
        nlc.temp_values = np.array([1.0])
        
        ps_array = [np.array([s]) for s in partial_sums]
        
        wynn_result, wynn_err = nlc.wynn_epsilon(ps_array)
        wynn_val = wynn_result[0]
        
        euler_result, euler_err = nlc.euler_resummation(ps_array, l=3)
        euler_val = euler_result[0]
        
        theta_result, theta_err = nlc.brezinski_theta(ps_array)
        theta_val = theta_result[0]
        
        if verbose:
            print(f"\nWynn ε: {wynn_val:.10f}, error: {abs(wynn_val - exact_sum):.2e}")
            print(f"Euler:  {euler_val:.10f}, error: {abs(euler_val - exact_sum):.2e}")
            print(f"Theta:  {theta_val:.10f}, error: {abs(theta_val - exact_sum):.2e}")
        
        # This converges fast, so acceleration may not help much
        # Just check it doesn't break
        wynn_reasonable = abs(wynn_val - exact_sum) < 0.1
        
        if verbose:
            print(f"\n✓ Wynn gives reasonable result: {wynn_reasonable}")
        
        return {
            'wynn_error': abs(wynn_val - exact_sum),
            'euler_error': abs(euler_val - exact_sum),
            'theta_error': abs(theta_val - exact_sum),
            'direct_error': abs(partial_sums[-1] - exact_sum),
            'wynn_reasonable': wynn_reasonable
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print("RESUMMATION ALGORITHM VALIDATION TESTS")
    print("="*80)
    
    results = {}
    
    # Test 1: Geometric series (Wynn should excel)
    results['geometric'] = test_geometric_series(verbose=True)
    
    # Test 2: Alternating harmonic (Euler should excel)
    results['alternating'] = test_alternating_harmonic(verbose=True)
    
    # Test 3: Exponential (fast convergence, just check no blow-up)
    results['exponential'] = test_exponential_series(verbose=True)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    all_passed = True
    
    if results['geometric']:
        geom = results['geometric']
        if geom['wynn_better']:
            print("✓ Geometric series: Wynn acceleration working")
        else:
            print("✗ Geometric series: Wynn acceleration FAILED")
            all_passed = False
    else:
        print("✗ Geometric series: TEST CRASHED")
        all_passed = False
    
    if results['alternating']:
        alt = results['alternating']
        if alt['euler_better']:
            print("✓ Alternating series: Euler acceleration working")
        else:
            print("✗ Alternating series: Euler acceleration FAILED")
            all_passed = False
    else:
        print("✗ Alternating series: TEST CRASHED")
        all_passed = False
    
    if results['exponential']:
        exp = results['exponential']
        if exp['wynn_reasonable']:
            print("✓ Exponential series: No catastrophic blow-up")
        else:
            print("✗ Exponential series: Blow-up detected")
            all_passed = False
    else:
        print("✗ Exponential series: TEST CRASHED")
        all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓\n")
        return 0
    else:
        print("\n✗✗✗ SOME TESTS FAILED ✗✗✗\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
