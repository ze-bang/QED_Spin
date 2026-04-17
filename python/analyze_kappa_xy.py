#!/usr/bin/env python3
"""
Analyze the energy-current cross-correlator S_{JEx,JEy}(omega,T) and extract
the thermal Hall conductivity kappa_xy.

Reads the HDF5 output from the ED computation and plots:
1. S_{JEx,JEy}(omega) at each temperature (real and imaginary parts)
2. Im[S]/omega vs omega (relevant for kappa_xy extraction)
3. kappa_xy(T) from the omega -> 0 limit

Usage:
    python analyze_kappa_xy.py <output_dir>
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os


def load_dynamical_response(h5_path, operator_name=None):
    """Load dynamical response data from HDF5 file."""
    data = {}
    with h5py.File(h5_path, 'r') as f:
        if 'dynamical' in f:
            dyn = f['dynamical']
            for key in dyn.keys():
                grp = dyn[key]
                if 'frequencies' not in grp:
                    continue
                entry = {
                    'frequencies': np.array(grp['frequencies']),
                    'spectral_real': np.array(grp['spectral_real']),
                    'spectral_imag': np.array(grp['spectral_imag']),
                }
                if 'error_real' in grp:
                    entry['error_real'] = np.array(grp['error_real'])
                if 'error_imag' in grp:
                    entry['error_imag'] = np.array(grp['error_imag'])
                if 'temperature' in grp.attrs:
                    entry['temperature'] = grp.attrs['temperature']
                elif 'temperature' in grp:
                    entry['temperature'] = float(np.array(grp['temperature']))
                data[key] = entry
        
        # Also check for FTLM samples
        if 'ftlm_samples' in f:
            data['_ftlm_samples'] = True
    
    return data


def extract_kappa_xy(frequencies, spectral_imag, temperature, broadening=0.1):
    """Extract kappa_xy from Im[S_{JEx,JEy}(omega)] / omega.
    
    The thermal Hall conductivity is:
        kappa_xy = (1/T) * lim_{omega->0} Im[S(omega)] / omega
    
    In practice, we extract the low-frequency slope.
    """
    # Focus on small positive omega region
    omega_mask = (frequencies > broadening/2) & (frequencies < 5*broadening)
    
    if np.sum(omega_mask) < 3:
        return 0.0, 0.0
    
    omega_sel = frequencies[omega_mask]
    imag_sel = spectral_imag[omega_mask]
    
    # Im[S(omega)] / omega
    ratio = imag_sel / omega_sel
    
    # Average in the low-frequency window
    kappa = np.mean(ratio) / temperature
    kappa_err = np.std(ratio) / np.sqrt(len(ratio)) / temperature
    
    return kappa, kappa_err


def main():
    if len(sys.argv) < 2:
        output_dir = "./test_pyro16_qsi_kappa/output"
    else:
        output_dir = sys.argv[1]
    
    h5_path = os.path.join(output_dir, "ed_results.h5")
    
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found. Run the ED computation first.")
        sys.exit(1)
    
    # Print HDF5 structure
    print("HDF5 structure:")
    with h5py.File(h5_path, 'r') as f:
        def print_tree(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"  {name}/")
                for attr_name, attr_val in obj.attrs.items():
                    print(f"    @{attr_name} = {attr_val}")
        f.visititems(print_tree)
    
    # Load data
    data = load_dynamical_response(h5_path)
    
    if not data:
        print("No dynamical response data found in HDF5.")
        sys.exit(1)
    
    print(f"\nFound {len(data)} datasets")
    
    # Sort by temperature
    temp_data = []
    for key, entry in data.items():
        if key.startswith('_'):
            continue
        temp = entry.get('temperature', 0.0)
        temp_data.append((temp, key, entry))
    temp_data.sort(key=lambda x: x[0])
    
    # --- Plot 1: Spectral function S(omega) ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    for temp, key, entry in temp_data:
        freq = entry['frequencies']
        label = f"T={temp:.3f}" if temp > 0 else key
        axes[0].plot(freq, entry['spectral_real'], label=label, alpha=0.8)
        axes[1].plot(freq, entry['spectral_imag'], label=label, alpha=0.8)
    
    axes[0].set_ylabel(r"Re[$S_{J^E_x, J^E_y}(\omega)$]")
    axes[0].set_title("Energy current cross-correlator")
    axes[0].legend()
    axes[0].axhline(0, color='k', linewidth=0.5)
    
    axes[1].set_xlabel(r"$\omega / J_{zz}$")
    axes[1].set_ylabel(r"Im[$S_{J^E_x, J^E_y}(\omega)$]") 
    axes[1].legend()
    axes[1].axhline(0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "spectral_function_JE.png"), dpi=150)
    print(f"\nSaved: {os.path.join(output_dir, 'spectral_function_JE.png')}")
    
    # --- Plot 2: Im[S]/omega for kappa_xy extraction ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    temperatures_list = []
    kappa_list = []
    kappa_err_list = []
    
    for temp, key, entry in temp_data:
        freq = entry['frequencies']
        imag = entry['spectral_imag']
        
        # Compute Im[S]/omega (avoid division by zero)
        mask = np.abs(freq) > 0.01
        ratio = np.zeros_like(freq)
        ratio[mask] = imag[mask] / freq[mask]
        
        ax2.plot(freq, ratio, label=f"T={temp:.3f}", alpha=0.8)
        
        # Extract kappa_xy
        if temp > 0:
            kappa, kappa_err = extract_kappa_xy(freq, imag, temp)
            temperatures_list.append(temp)
            kappa_list.append(kappa)
            kappa_err_list.append(kappa_err)
            print(f"  T={temp:.4f}: kappa_xy = {kappa:.6e} +/- {kappa_err:.6e}")
    
    ax2.set_xlabel(r"$\omega / J_{zz}$")
    ax2.set_ylabel(r"Im[$S_{J^E_x, J^E_y}(\omega)$] / $\omega$")
    ax2.set_title(r"$\kappa_{xy}$ extraction: Im[$S$]/$\omega$")
    ax2.legend()
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_xlim(-1, 1)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "kappa_xy_extraction.png"), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'kappa_xy_extraction.png')}")
    
    # --- Plot 3: kappa_xy(T) ---
    if temperatures_list:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.errorbar(temperatures_list, kappa_list, yerr=kappa_err_list,
                     fmt='o-', capsize=4, markersize=6)
        ax3.set_xlabel(r"$T / J_{zz}$")
        ax3.set_ylabel(r"$\kappa_{xy}$ (arb. units)")
        ax3.set_title(r"Thermal Hall conductivity $\kappa_{xy}(T)$ — 16-site pyrochlore QSI")
        ax3.axhline(0, color='k', linewidth=0.5)
        
        plt.tight_layout()
        fig3.savefig(os.path.join(output_dir, "kappa_xy_temperature.png"), dpi=150)
        print(f"Saved: {os.path.join(output_dir, 'kappa_xy_temperature.png')}")
    
    # Save numerical data
    if temperatures_list:
        np.savez(os.path.join(output_dir, "kappa_xy_data.npz"),
                 temperatures=np.array(temperatures_list),
                 kappa_xy=np.array(kappa_list),
                 kappa_xy_err=np.array(kappa_err_list))
        print(f"Saved: {os.path.join(output_dir, 'kappa_xy_data.npz')}")
    
    plt.show()


if __name__ == "__main__":
    main()
