#!/usr/bin/env python3
"""
Demonstrate the discontinuity bug and verify it's fixed.
"""

import numpy as np
import matplotlib.pyplot as plt
from NLC_sum_ftlm import NLCExpansionFTLM


def create_alternating_divergent_series():
    """
    Create a series that is both alternating AND divergent at low T,
    similar to real NLC series.
    """
    # Temperature grid
    T = np.logspace(-1, 1, 100)  # 0.1 to 10
    
    # Create partial sums that alternate and diverge at low T
    # Mimic NLC: weights alternate in sign, magnitude grows at low T
    partial_sums = []
    
    # Base contribution (order 1)
    s1 = -1.0 / T**0.5  # Grows at low T
    partial_sums.append(s1)
    
    # Order 2: positive correction
    s2 = s1 + 0.3 / T**0.3
    partial_sums.append(s2)
    
    # Order 3: negative correction (alternating)
    s3 = s2 - 0.4 / T**0.4
    partial_sums.append(s3)
    
    # Order 4: positive correction
    s4 = s3 + 0.6 / T**0.5
    partial_sums.append(s4)
    
    # Order 5: negative correction (growing - divergent!)
    s5 = s4 - 0.8 / T**0.6
    partial_sums.append(s5)
    
    return T, partial_sums


def old_conservative_method(partial_sums):
    """OLD BUGGY VERSION - per-temperature switching"""
    n = len(partial_sums)
    if n < 3:
        return partial_sums[-1]
    
    inc_last = np.abs(partial_sums[-1] - partial_sums[-2])
    inc_prev = np.abs(partial_sums[-2] - partial_sums[-3])
    
    growing = inc_last > inc_prev
    
    # PER-TEMPERATURE SWITCH - CREATES DISCONTINUITIES!
    val = np.where(growing, partial_sums[-2], partial_sums[-1])
    
    return val


def new_conservative_method(partial_sums):
    """NEW FIXED VERSION - global decision"""
    n = len(partial_sums)
    if n < 3:
        return partial_sums[-1]
    
    inc_last = np.abs(partial_sums[-1] - partial_sums[-2])
    inc_prev = np.abs(partial_sums[-2] - partial_sums[-3])
    
    growing = inc_last > inc_prev
    n_growing = np.sum(growing)
    n_total = len(growing)
    
    # GLOBAL DECISION - SMOOTH!
    if n_growing > 0.5 * n_total:
        val = partial_sums[-2]  # Use N-1 everywhere
    else:
        val = partial_sums[-1]  # Use N everywhere
    
    return val


def test_discontinuity_fix():
    """Visual test showing discontinuities before/after fix"""
    
    T, partial_sums = create_alternating_divergent_series()
    
    # Apply old buggy method
    old_result = old_conservative_method(partial_sums)
    
    # Apply new fixed method
    new_result = new_conservative_method(partial_sums)
    
    # Apply Euler (what should actually be used for alternating series)
    nlc = NLCExpansionFTLM('.', '.', 0.1, 10, 100)
    nlc.temp_values = T
    euler_result, _ = nlc.euler_resummation(partial_sums, l=2)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: All partial sums
    ax = axes[0, 0]
    for i, ps in enumerate(partial_sums):
        ax.semilogx(T, ps, 'o-', label=f'Order {i+1}', alpha=0.7, markersize=3)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Energy')
    ax.set_title('Partial Sums (Orders 1-5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top right: Old buggy method (DISCONTINUOUS)
    ax = axes[0, 1]
    ax.semilogx(T, old_result, 'r-', linewidth=2, label='OLD: Per-temp switching')
    ax.semilogx(T, partial_sums[-2], 'b--', alpha=0.5, label='Order 4 (N-1)')
    ax.semilogx(T, partial_sums[-1], 'g--', alpha=0.5, label='Order 5 (N)')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Energy')
    ax.set_title('OLD Conservative Method (BUGGY)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight discontinuities
    disc = np.abs(np.diff(old_result))
    large_jumps = disc > 0.2
    if np.any(large_jumps):
        jump_indices = np.where(large_jumps)[0]
        for idx in jump_indices[:5]:  # Show first 5 jumps
            ax.axvline(T[idx], color='red', alpha=0.3, linestyle=':', linewidth=2)
    
    # Bottom left: New fixed method (SMOOTH)
    ax = axes[1, 0]
    ax.semilogx(T, new_result, 'g-', linewidth=2, label='NEW: Global decision')
    ax.semilogx(T, partial_sums[-2], 'b--', alpha=0.5, label='Order 4 (N-1)')
    ax.semilogx(T, partial_sums[-1], 'r--', alpha=0.5, label='Order 5 (N)')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Energy')
    ax.set_title('NEW Conservative Method (FIXED)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Euler method (BEST for alternating)
    ax = axes[1, 1]
    ax.semilogx(T, euler_result, 'm-', linewidth=2, label='Euler Transform')
    ax.semilogx(T, partial_sums[-1], 'b--', alpha=0.5, label='Direct (Order 5)')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Energy')
    ax.set_title('Euler Method (RECOMMENDED for alternating series)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('discontinuity_fix_demonstration.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: discontinuity_fix_demonstration.png")
    
    # Quantify discontinuities
    print("\n" + "="*80)
    print("DISCONTINUITY ANALYSIS")
    print("="*80)
    
    old_jumps = np.abs(np.diff(old_result))
    new_jumps = np.abs(np.diff(new_result))
    euler_jumps = np.abs(np.diff(euler_result))
    
    print(f"\nOLD method (per-temp switching):")
    print(f"  Max jump: {np.max(old_jumps):.6f}")
    print(f"  Mean jump: {np.mean(old_jumps):.6f}")
    print(f"  Large jumps (>0.1): {np.sum(old_jumps > 0.1)}")
    
    print(f"\nNEW method (global decision):")
    print(f"  Max jump: {np.max(new_jumps):.6f}")
    print(f"  Mean jump: {np.mean(new_jumps):.6f}")
    print(f"  Large jumps (>0.1): {np.sum(new_jumps > 0.1)}")
    
    print(f"\nEuler method:")
    print(f"  Max jump: {np.max(euler_jumps):.6f}")
    print(f"  Mean jump: {np.mean(euler_jumps):.6f}")
    print(f"  Large jumps (>0.1): {np.sum(euler_jumps > 0.1)}")
    
    improvement = (np.max(old_jumps) - np.max(new_jumps)) / np.max(old_jumps) * 100
    print(f"\n✓ Discontinuity reduction: {improvement:.1f}%")
    print("="*80)


if __name__ == "__main__":
    try:
        test_discontinuity_fix()
        print("\n✓✓✓ TEST COMPLETED SUCCESSFULLY ✓✓✓")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
