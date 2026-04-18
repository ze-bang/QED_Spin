#!/usr/bin/env python3
"""
Script to print the Gamma matrices for each sublattice index (sub_j = 0, 1, 2, 3)
"""

import numpy as np

# Phase factor: ω = e^(i*2π/3)
omega = np.exp(1j * 2 * np.pi / 3)

# Base Gamma matrix for sub_j = 0
Gamma_base = np.array([[0, 0, 0, 0],
                       [0, 1, 1, omega**2],
                       [0, 1, omega, omega],
                       [0, omega**2, omega, omega**2]], dtype=complex)

print("="*70)
print("Gamma Matrices for Non-Kramers Doublet Three-Spin Interaction")
print("="*70)
print(f"\nPhase factor: ω = e^(i*2π/3) = {omega:.6f}")
print(f"              ω² = e^(i*4π/3) = {omega**2:.6f}")
print("\n" + "="*70)

for sub_j in range(4):
    print(f"\nSublattice index sub_j = {sub_j}")
    print("-"*70)
    
    # Permute rows and columns based on sub_j
    # Create permutation: shift indices by sub_j (cyclic)
    perm = np.array([(idx - sub_j) % 4 for idx in range(4)])
    Gamma = Gamma_base[np.ix_(perm, perm)]
    
    print(f"Permutation: {perm}")
    print(f"\nGamma[{sub_j}] =")
    
    # Print the matrix with proper formatting
    for i in range(4):
        row_str = "  ["
        for j in range(4):
            val = Gamma[i, j]
            if np.abs(val) < 1e-10:
                row_str += "      0      "
            elif np.abs(val - 1) < 1e-10:
                row_str += "      1      "
            elif np.abs(val - omega) < 1e-10:
                row_str += "      ω      "
            elif np.abs(val - omega**2) < 1e-10:
                row_str += "      ω²     "
            else:
                row_str += f"{val:12.6f}"
            
            if j < 3:
                row_str += ", "
        row_str += "]"
        print(row_str)
    
    # Also print numeric values
    print(f"\nNumeric representation:")
    for i in range(4):
        row_str = "  ["
        for j in range(4):
            val = Gamma[i, j]
            row_str += f"{val.real:7.4f}{val.imag:+7.4f}j"
            if j < 3:
                row_str += ", "
        row_str += "]"
        print(row_str)
    
    print()

print("="*70)
