#!/usr/bin/env python3
"""
Simple script to prune NdMgAl11O19 heat capacity data for NLCE fitting.
Outputs a two-column file: T(K) C(J/mol/K)
"""
import numpy as np
import os

# Input file
data_file = "/home/pc_linux/exact_diagonalization_clean/20211010_LFZR_NdMgAl11O19_HC_0.80mg_1(1).dat"

# Read the file with latin-1 encoding (handles µ symbol)
with open(data_file, 'r', encoding='latin-1') as f:
    lines = f.readlines()

# Find the data section
data_start = None
for i, line in enumerate(lines):
    if line.startswith('[Data]'):
        data_start = i + 2  # Skip [Data] and header line
        break

if data_start is None:
    raise ValueError("Could not find [Data] section")

# Parse data
temp = []
spec_heat = []
spec_heat_err = []

for line in lines[data_start:]:
    if not line.strip():
        continue
    parts = line.strip().split(',')
    if len(parts) < 11:
        continue
    try:
        T = float(parts[7])  # Sample Temp (Kelvin)
        C = float(parts[9])  # Samp HC (µJ/mg-K)
        C_err = float(parts[10])  # Samp HC Err (µJ/mg-K)
        temp.append(T)
        spec_heat.append(C)
        spec_heat_err.append(C_err)
    except (ValueError, IndexError):
        continue

temp = np.array(temp)
spec_heat = np.array(spec_heat)
spec_heat_err = np.array(spec_heat_err)

# Convert to SI units: µJ/(mg·K) → J/(mol·K)
# Formula weight: 769.33 g/mol
formula_weight = 769.33
spec_heat_SI = spec_heat * 1e-3 * formula_weight
spec_heat_err_SI = spec_heat_err * 1e-3 * formula_weight

# Prune: remove high-T data (T > 2.5 K) and large error points
mask = (temp <= 2.5) & (np.abs(spec_heat_err / spec_heat) < 0.5)
temp_pruned = temp[mask]
spec_heat_SI_pruned = spec_heat_SI[mask]

# Sort by temperature
sort_idx = np.argsort(temp_pruned)
temp_pruned = temp_pruned[sort_idx]
spec_heat_SI_pruned = spec_heat_SI_pruned[sort_idx]

print(f"Loaded {len(temp)} points, kept {len(temp_pruned)} after pruning")
print(f"Temperature range: {temp_pruned.min():.3f} K to {temp_pruned.max():.3f} K")

# Save for NLCE fitting (just T and C columns)
output_file = "/home/pc_linux/exact_diagonalization_clean/NdMgAl11O19_for_nlce.dat"
np.savetxt(output_file, np.column_stack([temp_pruned, spec_heat_SI_pruned]),
           header="T(K) C(J/mol/K)", fmt='%.6e')
print(f"Saved to: {output_file}")

# Also plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.plot(temp_pruned, spec_heat_SI_pruned, 'o-', markersize=4)
plt.xlabel('Temperature (K)')
plt.ylabel('Specific Heat (J/mol/K)')
plt.title('NdMgAl$_{11}$O$_{19}$ - Pruned for NLCE fitting')
plt.grid(True, alpha=0.3)
plt.savefig('/home/pc_linux/exact_diagonalization_clean/NdMgAl11O19_pruned_for_nlce.png', dpi=150)
print("Plot saved")
