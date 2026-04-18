#!/usr/bin/env python3
"""
Plot and analyze NdMgAl11O19 specific heat data.
Data shows a cusp around 1K characteristic of frustrated magnetism.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_ppms_data(filepath):
    """
    Load PPMS heat capacity data file.
    
    Returns:
        temp: Temperature in Kelvin
        spec_heat: Specific heat in µJ/(mg·K)
        spec_heat_err: Error in specific heat
    """
    data = []
    in_data_section = False
    
    # Try different encodings
    for encoding in ['latin-1', 'cp1252', 'utf-8']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line == '[Data]':
                        in_data_section = True
                        # Skip header line
                        next(f)
                        continue
                    
                    if in_data_section and line:
                        parts = line.split(',')
                        if len(parts) >= 11:
                            try:
                                # Column 7: Sample Temp (Kelvin)
                                # Column 9: Samp HC (µJ/mg-K)
                                # Column 10: Samp HC Err (µJ/mg-K)
                                temp = float(parts[7])
                                spec_heat = float(parts[9])
                                spec_heat_err = float(parts[10])
                                data.append([temp, spec_heat, spec_heat_err])
                            except (ValueError, IndexError):
                                continue
            break  # Success
        except UnicodeDecodeError:
            continue
    
    data = np.array(data)
    # Sort by temperature
    sort_idx = np.argsort(data[:, 0])
    data = data[sort_idx]
    
    return data[:, 0], data[:, 1], data[:, 2]


def convert_to_SI(spec_heat_uj_mg_K, formula_weight=769.33, atoms_per_fu=32):
    """
    Convert from µJ/(mg·K) to J/(mol·K).
    
    Args:
        spec_heat_uj_mg_K: Specific heat in µJ/(mg·K)
        formula_weight: Molecular weight in g/mol (769.33 for NdMgAl11O19)
        atoms_per_fu: Atoms per formula unit (32 for NdMgAl11O19)
    
    Returns:
        Specific heat in J/(mol·K)
    """
    # µJ/(mg·K) → J/(g·K): multiply by 1e-3
    # J/(g·K) → J/(mol·K): multiply by formula_weight
    return spec_heat_uj_mg_K * 1e-3 * formula_weight


def convert_to_SI_per_spin(spec_heat_uj_mg_K, formula_weight=769.33, n_spins=1):
    """
    Convert from µJ/(mg·K) to J/(mol·K) per magnetic ion.
    
    For NdMgAl11O19: 1 Nd³⁺ per formula unit (S=9/2 for 4f³ configuration)
    
    Args:
        spec_heat_uj_mg_K: Specific heat in µJ/(mg·K)
        formula_weight: Molecular weight in g/mol
        n_spins: Number of magnetic ions per formula unit
    
    Returns:
        Specific heat in J/(mol_spin·K)
    """
    # Convert to per formula unit first
    C_per_fu = spec_heat_uj_mg_K * 1e-3 * formula_weight
    # Then per spin
    return C_per_fu / n_spins


def main():
    # Data file path
    data_file = "/home/pc_linux/exact_diagonalization_clean/20211010_LFZR_NdMgAl11O19_HC_0.80mg_1(1).dat"
    
    # Load data
    print("Loading PPMS heat capacity data...")
    temp, spec_heat, spec_heat_err = load_ppms_data(data_file)
    
    print(f"Loaded {len(temp)} data points")
    print(f"Temperature range: {temp.min():.3f} K to {temp.max():.3f} K")
    print(f"Specific heat range: {spec_heat.min():.4f} to {spec_heat.max():.4f} µJ/(mg·K)")
    
    # Convert to SI units (J/(mol·K) per formula unit)
    formula_weight = 769.33  # g/mol for NdMgAl11O19
    spec_heat_SI = convert_to_SI(spec_heat, formula_weight)
    spec_heat_err_SI = convert_to_SI(spec_heat_err, formula_weight)
    
    # Also compute per Nd spin (1 Nd per formula unit)
    spec_heat_per_spin = spec_heat_SI  # Already per mol of Nd since 1 Nd per f.u.
    
    # Prune data: remove points with very large relative errors (> 50%)
    rel_err = np.abs(spec_heat_err / spec_heat)
    mask = rel_err < 0.5
    
    temp_pruned = temp[mask]
    spec_heat_pruned = spec_heat[mask]
    spec_heat_err_pruned = spec_heat_err[mask]
    spec_heat_SI_pruned = spec_heat_SI[mask]
    spec_heat_err_SI_pruned = spec_heat_err_SI[mask]
    
    print(f"\nAfter pruning: {len(temp_pruned)} data points")
    print(f"Temperature range: {temp_pruned.min():.3f} K to {temp_pruned.max():.3f} K")
    
    # Report data characteristics
    print(f"\n*** Data characteristics ***")
    print(f"    Low T maximum: C = {spec_heat_SI_pruned.max():.4f} J/(mol·K) at T = {temp_pruned[np.argmax(spec_heat_SI_pruned)]:.3f} K")
    print(f"    Data suitable for fitting magnetic Schottky anomaly or spin model.")
    
    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Raw data (µJ/(mg·K)) - linear scale
    ax = axes[0, 0]
    ax.errorbar(temp_pruned, spec_heat_pruned, yerr=spec_heat_err_pruned, 
                fmt='o', markersize=4, capsize=2, alpha=0.7, label='Data')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Specific Heat (µJ/(mg·K))')
    ax.set_title('NdMgAl₁₁O₁₉ Heat Capacity (linear scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    
    # Panel 2: Raw data - log scale
    ax = axes[0, 1]
    ax.errorbar(temp_pruned, spec_heat_pruned, yerr=spec_heat_err_pruned, 
                fmt='o', markersize=4, capsize=2, alpha=0.7, label='Data')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Specific Heat (µJ/(mg·K))')
    ax.set_title('NdMgAl₁₁O₁₉ Heat Capacity (log scale)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    # Panel 3: SI units J/(mol·K) - linear scale
    ax = axes[1, 0]
    ax.errorbar(temp_pruned, spec_heat_SI_pruned, yerr=spec_heat_err_SI_pruned, 
                fmt='s', markersize=4, capsize=2, alpha=0.7, color='green', label='Data')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Specific Heat (J/(mol·K))')
    ax.set_title('NdMgAl₁₁O₁₉ Heat Capacity - SI Units')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 2.5)
    
    # Panel 4: C/T vs T (useful for seeing low-T behavior)
    ax = axes[1, 1]
    C_over_T = spec_heat_SI_pruned / temp_pruned
    C_over_T_err = spec_heat_err_SI_pruned / temp_pruned
    ax.errorbar(temp_pruned, C_over_T, yerr=C_over_T_err, 
                fmt='^', markersize=4, capsize=2, alpha=0.7, color='purple', label='C/T')
    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('C/T (J/(mol·K²))')
    ax.set_title('NdMgAl₁₁O₁₉: C/T vs T')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(data_file)
    output_file = os.path.join(output_dir, 'NdMgAl11O19_specific_heat.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save pruned data for NLCE fitting
    # Format: T(K)  C(J/mol/K)  C_err(J/mol/K)
    output_data_file = os.path.join(output_dir, 'NdMgAl11O19_specific_heat_pruned.txt')
    header = "# NdMgAl11O19 specific heat data (pruned)\n"
    header += "# Formula weight: 769.33 g/mol, 1 Nd per formula unit\n"
    header += "# Columns: Temperature(K)  C_SI(J/mol/K)  C_err_SI(J/mol/K)\n"
    
    data_out = np.column_stack([temp_pruned, spec_heat_SI_pruned, spec_heat_err_SI_pruned])
    np.savetxt(output_data_file, data_out, header=header, 
               fmt='%.6e', comments='')
    print(f"Pruned data saved to: {output_data_file}")
    
    # Show plot
    plt.show()
    
    return temp_pruned, spec_heat_SI_pruned


if __name__ == '__main__':
    temp, spec_heat = main()
