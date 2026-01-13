#!/usr/bin/env python3
"""
Plot static structure factor S(q) vs temperature from TPQ_DSSF results.
Compares with compute_bfg_order_parameters.cpp output.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import os
import subprocess

def load_sssf_results(h5_path):
    """Load SSSF results from unified HDF5 file."""
    results = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Check for 'static' group (SSSF data is stored here)
        if 'static' not in f:
            print("No static data found in HDF5 file")
            return results
        
        static_group = f['static']
        
        for obs_name in static_group.keys():
            obs_group = static_group[obs_name]
            
            # Collect all temperatures and values
            temps = []
            values = []
            
            for sample_key in obs_group.keys():
                sample_group = obs_group[sample_key]
                if 'temperatures' in sample_group and 'expectation' in sample_group:
                    T = sample_group['temperatures'][:]
                    S = sample_group['expectation'][:]
                    for t, s in zip(T, S):
                        temps.append(t)
                        values.append(s)
            
            if temps:
                # Sort by temperature
                sorted_idx = np.argsort(temps)
                results[obs_name] = {
                    'T': np.array(temps)[sorted_idx],
                    'S': np.array(values)[sorted_idx]
                }
    
    return results

def load_bfg_order_params(h5_path):
    """Load S(q) from compute_bfg_order_parameters output."""
    results = {}
    
    if not os.path.exists(h5_path):
        return results
    
    with h5py.File(h5_path, 'r') as f:
        # Data is at root level: k_points, S_q (complex), s_q_real
        if 'k_points' in f and 'S_q' in f:
            k_points = f['k_points'][:]
            s_q = f['S_q'][:]  # Complex values
            s_q_real = np.real(s_q)  # Take real part
            
            # Find Gamma and M points with flexible matching
            for i, (kx, ky) in enumerate(k_points):
                # Gamma: (0,0)
                if abs(kx) < 0.01 and abs(ky) < 0.01:
                    results['Gamma'] = s_q_real[i]
                # M point: approximately (π, π) - but in kagome may have different ky
                # Look for kx ≈ π and ky ≈ 3.02 (closest to π in this basis)
                elif abs(kx - np.pi) < 0.1 and abs(ky - 3.02) < 0.1:
                    results['M'] = s_q_real[i]
                # Also try exact (π, π) matching
                elif abs(kx - np.pi) < 0.1 and abs(ky - np.pi) < 0.2:
                    if 'M' not in results:
                        results['M'] = s_q_real[i]
        
        # Also check for spin_structure_factor group (older format)
        elif 'spin_structure_factor' in f:
            sf_group = f['spin_structure_factor']
            
            if 'k_points' in sf_group and 's_q_real' in sf_group:
                k_points = sf_group['k_points'][:]
                s_q_real = sf_group['s_q_real'][:]
                
                for i, (kx, ky) in enumerate(k_points):
                    if abs(kx) < 0.01 and abs(ky) < 0.01:
                        results['Gamma'] = s_q_real[i]
                    elif abs(kx - np.pi) < 0.1 and abs(ky - np.pi) < 0.2:
                        results['M'] = s_q_real[i]
    
    return results

def main():
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_kagome_2x3'
    h5_path = os.path.join(cluster_dir, 'structure_factor_results', 'dssf_results.h5')
    
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found")
        return
    
    print(f"Loading SSSF results from {h5_path}")
    results = load_sssf_results(h5_path)
    
    if not results:
        print("No results to plot")
        return
    
    print(f"Found {len(results)} observables:")
    for name in results.keys():
        print(f"  - {name}: {len(results[name]['T'])} data points")
    
    # Find Gamma and M point data (SmSp = S-S+ which matches "0,0")
    gamma_key = None
    m_key = None
    
    for key in results.keys():
        if 'SmSp' in key:
            # Gamma point: Qx0_Qy0_Qz0 (all zeros)
            if 'Qx0_Qy0_Qz0' in key:
                gamma_key = key
            # M point: Qx3.14159_Qy3.14159
            elif 'Qx3.14159_Qy3.14159' in key:
                m_key = key
    
    if gamma_key is None:
        # Fallback: find key with most data points that looks like Gamma
        for key in results.keys():
            if 'SmSp' in key and 'Qz0' in key:
                if gamma_key is None or len(results[key]['T']) > len(results[gamma_key]['T']):
                    if 'Qx0' in key and 'Qy0' in key:
                        gamma_key = key
    
    if m_key is None:
        # Try to find any M point data
        for key in results.keys():
            if key != gamma_key:
                m_key = key
                break
    
    print(f"\nPlotting:")
    print(f"  Gamma point: {gamma_key}")
    print(f"  M point: {m_key}")
    
    # Load compute_bfg_order_parameters results for comparison
    bfg_h5_path = os.path.join(cluster_dir, 'structure_factor_results', 'order_parameters.h5')
    bfg_results = load_bfg_order_params(bfg_h5_path)
    
    if bfg_results:
        print(f"\nLoaded compute_bfg_order_parameters results:")
        for k, v in bfg_results.items():
            print(f"  S({k}) = {v:.6f}")
    else:
        print(f"\nNo compute_bfg_order_parameters results found at {bfg_h5_path}")
        print("Running compute_bfg_order_parameters to generate comparison data...")
        
        # Find wavefunction file
        wf_file = None
        for candidate in ['ground_state.h5', 'wavefunction.h5', 'tpq_states.h5']:
            path = os.path.join(cluster_dir, 'output', candidate)
            if os.path.exists(path):
                wf_file = path
                break
        
        if wf_file:
            # Run compute_bfg_order_parameters
            bfg_exe = './build/compute_bfg_order_parameters'
            if os.path.exists(bfg_exe):
                cmd = [bfg_exe, wf_file, cluster_dir, bfg_h5_path]
                print(f"  Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    bfg_results = load_bfg_order_params(bfg_h5_path)
                    print(f"  Generated results:")
                    for k, v in bfg_results.items():
                        print(f"    S({k}) = {v:.6f}")
                else:
                    print(f"  Error running compute_bfg_order_parameters: {result.stderr}")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot S(q) vs T
    ax1 = axes[0]
    
    if gamma_key and gamma_key in results:
        data = results[gamma_key]
        ax1.scatter(data['T'], data['S'], s=80, marker='o', color='blue', 
                   edgecolors='black', linewidth=0.5, label=r"TPQ_DSSF '0,0' $\Gamma$", zorder=5)
        ax1.plot(data['T'], data['S'], 'b-', alpha=0.5, linewidth=1)
    
    if m_key and m_key in results:
        data = results[m_key]
        ax1.scatter(data['T'], data['S'], s=80, marker='s', color='red',
                   edgecolors='black', linewidth=0.5, label=r"TPQ_DSSF '0,0' M", zorder=5)
        ax1.plot(data['T'], data['S'], 'r-', alpha=0.5, linewidth=1)
    
    # Add compute_bfg_order_parameters results as horizontal lines (ground state)
    if 'Gamma' in bfg_results:
        ax1.axhline(y=bfg_results['Gamma'], color='blue', linestyle='--', linewidth=2,
                   label=f"BFG $\\Gamma$ (T=0): {bfg_results['Gamma']:.4f}")
    if 'M' in bfg_results:
        ax1.axhline(y=bfg_results['M'], color='red', linestyle='--', linewidth=2,
                   label=f"BFG M (T=0): {bfg_results['M']:.4f}")
    
    ax1.set_xlabel('Temperature T', fontsize=12)
    ax1.set_ylabel(r'$S(\mathbf{q}) = \langle S^-(-\mathbf{q}) S^+(\mathbf{q}) \rangle$', fontsize=12)
    ax1.set_title(r"Static Structure Factor Comparison", fontsize=13)
    ax1.legend(fontsize=10, loc='best')
    ax1.set_xlim(left=0)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for low-T limit comparison
    if gamma_key and gamma_key in results and 'Gamma' in bfg_results:
        data = results[gamma_key]
        low_T_idx = np.argmin(data['T'])
        tpq_val = data['S'][low_T_idx]
        bfg_val = bfg_results['Gamma']
        diff = abs(tpq_val - bfg_val)
        ax1.annotate(f"TPQ(T→0): {tpq_val:.4f}\nBFG: {bfg_val:.4f}\nΔ: {diff:.2e}", 
                    xy=(data['T'][low_T_idx], tpq_val),
                    xytext=(0.6, 0.85), textcoords='axes fraction',
                    fontsize=9, color='blue',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot S(q) vs 1/T (beta) on log scale
    ax2 = axes[1]
    
    if gamma_key and gamma_key in results:
        data = results[gamma_key]
        beta = 1.0 / data['T']
        ax2.semilogx(beta, data['S'], 'bo-', markersize=8, label=r"TPQ $\Gamma$")
    
    if m_key and m_key in results:
        data = results[m_key]
        beta = 1.0 / data['T']
        ax2.semilogx(beta, data['S'], 'rs-', markersize=8, label=r"TPQ M")
    
    # Add BFG results as horizontal lines
    if 'Gamma' in bfg_results:
        ax2.axhline(y=bfg_results['Gamma'], color='blue', linestyle='--', linewidth=2,
                   label=f"BFG $\\Gamma$")
    if 'M' in bfg_results:
        ax2.axhline(y=bfg_results['M'], color='red', linestyle='--', linewidth=2,
                   label=f"BFG M")
    
    ax2.set_xlabel(r'$\beta = 1/T$', fontsize=12)
    ax2.set_ylabel(r'$S(\mathbf{q})$', fontsize=12)
    ax2.set_title('Structure Factor vs Inverse Temperature', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(cluster_dir, 'structure_factor_results', 'sssf_vs_temperature.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    main()
