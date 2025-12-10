#!/usr/bin/env python3
"""
Test script to verify consistency of all spectral function methods:
1. FTLM (Finite Temperature Lanczos Method) - thermal averaging
2. Ground state + continued fraction (T=0 optimal)
3. Ground state + spectral from eigendecomposition (T=0 validation)

Uses 8-site pyrochlore cluster for testing (small enough for exact comparison).

The methods should agree at T→0 (β→∞ limit) within numerical precision.
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import sys
import glob
import re
from pathlib import Path

# Set the working directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
TEST_DIR = PROJECT_DIR / "test_spectral_consistency"

def setup_test_cluster():
    """Generate 8-site pyrochlore cluster with Ising-like Hamiltonian"""
    # Parameters: Jxx Jyy Jzz h fieldx fieldy fieldz output_dir dim1 dim2 dim3 pbc non_kramer
    # Use simple Ising model for clear spectral features
    cmd = [
        "python3", str(PROJECT_DIR / "python/edlib/helper_pyrochlore.py"),
        "0.0", "0.0", "1.0",  # Jxx, Jyy, Jzz (pure Ising)
        "0.1",  # Small field to lift degeneracy
        "0", "0", "1",  # Field direction (z)
        str(TEST_DIR),  # Output directory
        "1", "1", "2",  # Dimensions (1x1x2 = 8 sites)
        "1",  # PBC
        "0"   # Kramer doublet
    ]
    
    print("=" * 60)
    print("Setting up 8-site pyrochlore test cluster")
    print("=" * 60)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR))
    if result.returncode != 0:
        print(f"Error setting up cluster: {result.stderr}")
        return False
    print(result.stdout)
    return True


def run_ground_state_diagonalization():
    """Run full diagonalization to get ground state and compute all eigenvalues"""
    print("\n" + "=" * 60)
    print("Running ground state diagonalization")
    print("=" * 60)
    
    # Use the ED executable
    ed_exec = PROJECT_DIR / "build/ED"
    if not ed_exec.exists():
        print(f"Error: ED executable not found at {ed_exec}")
        return False
    
    # Run FULL diagonalization with eigenvectors - need all eigenvalues for spectral reference
    cmd = [
        str(ed_exec),
        str(TEST_DIR),
        "--method=FULL",
        "--eigenvectors",
        "--eigenvalues=FULL",  # Get all eigenvalues for reference
        f"--output={TEST_DIR}/output"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR))
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    # TPQ_DSSF expects eigenvectors in output/eigenvectors/ subdirectory
    # Create the directory structure and copy files
    eigenvectors_dir = TEST_DIR / "output/eigenvectors"
    eigenvectors_dir.mkdir(exist_ok=True)
    
    import shutil
    
    # Copy ground state eigenvector to expected location
    gs_source = TEST_DIR / "output/eigenvector_0.dat"
    gs_dest = eigenvectors_dir / "eigenvector_0.dat"
    if gs_source.exists() and not gs_dest.exists():
        shutil.copy2(gs_source, gs_dest)
        print(f"Copied ground state eigenvector to: {gs_dest}")
    
    # Also copy eigenvalues.dat 
    eig_source = TEST_DIR / "output/eigenvalues.dat"
    eig_dest = eigenvectors_dir / "eigenvalues.dat"
    if eig_source.exists() and not eig_dest.exists():
        shutil.copy2(eig_source, eig_dest)
        print(f"Copied eigenvalues to: {eig_dest}")
    
    return True


def run_ftlm_spectral(temperature=0.01):
    """
    Run FTLM spectral method (Method 1)
    
    This computes S(q,ω) using finite-temperature Lanczos with random sampling.
    At low temperature, should converge to ground state result.
    
    Uses 'spectral_thermal' mode which computes:
    S(q,ω) = (1/Z) Σ_r ⟨r|e^{-β H} O†(ω) O|r⟩
    
    At low T, this should match the ground state result.
    """
    print("\n" + "=" * 60)
    print(f"Running FTLM spectral_thermal method (T={temperature})")
    print("=" * 60)
    
    tpq_exec = PROJECT_DIR / "build/TPQ_DSSF"
    if not tpq_exec.exists():
        print(f"Error: TPQ_DSSF executable not found at {tpq_exec}")
        return None
    
    # For FTLM, we need random samples. Create a tpq_state file
    # Or use the spectral_thermal mode which handles this
    
    # TPQ_DSSF arguments for spectral_thermal:
    # <directory> <krylov_dim> <spin_combinations> spectral_thermal [operator_type] [basis] 
    # [omega_min,omega_max,omega_bins,broadening,beta] [unit_cell_size] [momentum_points]
    
    beta = 1.0 / temperature  # Very low temperature (high beta)
    
    # Create output directory separate from single-state spectral
    ftlm_output_dir = TEST_DIR / "ftlm_spectral_results"
    ftlm_output_dir.mkdir(exist_ok=True)
    
    cmd = [
        "mpirun", "-np", "1",  # Single process for small system
        str(tpq_exec),
        str(TEST_DIR),
        "100",  # Krylov dimension
        "2,2",  # SzSz correlation
        "spectral_thermal",  # FTLM thermal spectral method
        "sum",  # Sum operator type
        "ladder",  # Ladder basis
        f"-5.0,5.0,500,0.1,{beta}",  # omega_min, omega_max, bins, broadening, beta
        "4",  # unit cell size
        "0,0,0",  # Gamma point momentum
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TEST_DIR))
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"Warning (FTLM): {result.stderr}")
    
    # The output will be in TEST_DIR/structure_factor_results/
    # Copy to separate location for comparison
    import shutil
    src_dir = TEST_DIR / "structure_factor_results"
    if src_dir.exists():
        shutil.copytree(src_dir, ftlm_output_dir, dirs_exist_ok=True)
        print(f"Copied FTLM results to: {ftlm_output_dir}")
        # Clean up original for next method
        shutil.rmtree(src_dir)
    
    return ftlm_output_dir


def run_ground_state_dssf_continued_fraction():
    """
    Run ground state DSSF using continued fraction (Method 2)
    
    This is the optimal method for T=0 dynamics - uses continued fraction
    representation which is O(M) per frequency point.
    """
    print("\n" + "=" * 60)
    print("Running ground state DSSF (continued fraction method)")
    print("=" * 60)
    
    ed_exec = PROJECT_DIR / "build/ED"
    if not ed_exec.exists():
        print(f"Error: ED executable not found at {ed_exec}")
        return None
    
    output_dir = TEST_DIR / "gs_dssf_cf"
    output_dir.mkdir(exist_ok=True)
    
    cmd = [
        str(ed_exec),
        str(TEST_DIR),
        "--method=FULL",
        "--eigenvectors",
        "--ground-state-dssf",
        f"--output={output_dir}",
        "--dyn-krylov=100",
        "--dyn-omega-min=-5.0",
        "--dyn-omega-max=5.0",
        "--dyn-omega-points=500",
        "--dyn-broadening=0.1",
        "--dyn-spin-combinations=2,2",
        "--dyn-momentum-points=0,0,0",
        "--dyn-operator-type=sum",
        "--dyn-basis=ladder",
        "--dyn-unit-cell-size=4"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_DIR))
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"Warning (GS DSSF CF): {result.stderr}")
    
    return output_dir


def run_spectral_from_state():
    """
    Run single-state spectral method via TPQ_DSSF "spectral" mode (Method 3)
    
    This uses the Lanczos method applied to O|ψ⟩ and computes spectral function.
    Should give same result as continued fraction when using the same parameters.
    """
    print("\n" + "=" * 60)
    print("Running spectral from state via TPQ_DSSF")
    print("=" * 60)
    
    tpq_exec = PROJECT_DIR / "build/TPQ_DSSF"
    if not tpq_exec.exists():
        print(f"Error: TPQ_DSSF executable not found at {tpq_exec}")
        return None
    
    # Check that ground state eigenvector exists
    gs_file = TEST_DIR / "output/eigenvectors/eigenvector_0.dat"
    if not gs_file.exists():
        print(f"Error: Ground state eigenvector not found at {gs_file}")
        return None
    
    # Use 'spectral' (single state) method - this should use the ground state eigenvector
    cmd = [
        "mpirun", "-np", "1",
        str(tpq_exec),
        str(TEST_DIR),
        "100",  # Krylov dimension  
        "2,2",  # SzSz correlation
        "spectral",  # Single state spectral (not thermal)
        "sum",  # Sum operator type
        "ladder",  # Ladder basis
        "-5.0,5.0,500,0.1",  # omega_min, omega_max, bins, broadening
        "4",  # unit cell size
        "0,0,0",  # Gamma point momentum
    ]
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(TEST_DIR))
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print(f"Warning (Spectral State): {result.stderr}")
    
    # Output will be in TEST_DIR/structure_factor_results/
    return TEST_DIR / "structure_factor_results"


def find_spectral_files(directory):
    """Find spectral output files in a directory"""
    if not directory or not Path(directory).exists():
        return []
    
    # Look for common spectral file patterns
    patterns = [
        "*.txt",
        "*spectral*.txt",
        "*dssf*.txt"
    ]
    
    files = []
    for pattern in patterns:
        found = list(Path(directory).rglob(pattern))
        files.extend(found)
    
    return list(set(files))


def load_spectral_data(filepath):
    """Load spectral function data from file"""
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        omega = float(parts[0])
                        spectral = float(parts[1])
                        data.append([omega, spectral])
                    except ValueError:
                        continue
        if data:
            return np.array(data)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None


def compare_results():
    """Compare results from all three methods"""
    print("\n" + "=" * 60)
    print("Comparing spectral function results")
    print("=" * 60)
    
    results = {}
    
    # Look for files in each output directory
    dirs = {
        "FTLM_Spectral_Thermal": TEST_DIR / "ftlm_spectral_results",
        "TPQ_DSSF_Spectral": TEST_DIR / "structure_factor_results",
        "GS_DSSF_CF": TEST_DIR / "gs_dssf_cf/ground_state_dssf",
        "Exact_Reference": TEST_DIR / "exact_reference"
    }
    
    # Also check the main test dir and subdirectories
    for name, d in dirs.items():
        print(f"\nSearching in {d}...")
        if d.exists():
            files = find_spectral_files(d)
            if files:
                print(f"  Found {len(files)} files")
                for f in files[:10]:  # Show first 10
                    print(f"    - {f.name}")
                # Load first suitable file
                for f in files:
                    # Skip non-spectral files
                    if "sublattice" in f.name.lower() or "density" in f.name.lower():
                        continue
                    data = load_spectral_data(f)
                    if data is not None and len(data) > 10:
                        results[name] = {
                            'file': f,
                            'data': data
                        }
                        print(f"  Loaded: {f.name} ({len(data)} points)")
                        break
        else:
            print(f"  Directory not found")
    
    # Also check main test directory and beta_inf subdirectory
    additional_dirs = [
        TEST_DIR / "structure_factor_results/beta_inf",
        TEST_DIR
    ]
    
    for d in additional_dirs:
        if d.exists():
            files = find_spectral_files(d)
            if files:
                print(f"\nFiles in {d}:")
                for f in files[:10]:
                    print(f"  - {f.name}")
    
    if len(results) < 1:
        print("\nNo spectral data found for comparison.")
        print("Check if the methods produced output files.")
        return
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1 = axes[0]
    colors = {
        'TPQ_DSSF_Spectral': 'blue', 
        'GS_DSSF_CF': 'red', 
        'Exact_Reference': 'green',
        'Spectral_State': 'orange'
    }
    linestyles = {
        'TPQ_DSSF_Spectral': '-', 
        'GS_DSSF_CF': '--', 
        'Exact_Reference': ':',
        'Spectral_State': '-.'
    }
    
    for name, result in results.items():
        data = result['data']
        omega = data[:, 0]
        spectral = data[:, 1]
        ax1.plot(omega, spectral, label=f"{name} ({result['file'].name})", 
                color=colors.get(name, 'black'), 
                linestyle=linestyles.get(name, '-'),
                alpha=0.8, linewidth=1.5)
    
    ax1.set_xlabel('ω', fontsize=12)
    ax1.set_ylabel('S(q,ω)', fontsize=12)
    ax1.set_title('Spectral Function Comparison - 8-site Pyrochlore', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Compute differences if we have multiple results
    ax2 = axes[1]
    
    if len(results) >= 2:
        # Use first result as reference
        names_list = list(results.keys())
        ref_name = names_list[0]
        ref_data = results[ref_name]['data']
        ref_omega = ref_data[:, 0]
        ref_spectral = ref_data[:, 1]
        
        for name in names_list[1:]:
            result = results[name]
            data = result['data']
            omega = data[:, 0]
            spectral = data[:, 1]
            
            # Simple comparison if grids match
            if len(omega) == len(ref_omega) and np.allclose(omega, ref_omega, rtol=1e-3):
                diff = spectral - ref_spectral
                rel_diff = np.abs(diff) / (np.abs(ref_spectral) + 1e-10)
                ax2.plot(omega, diff, label=f'{name} - {ref_name}', 
                        color=colors.get(name, 'black'), alpha=0.8)
                
                # Print statistics
                max_diff = np.max(np.abs(diff))
                mean_diff = np.mean(np.abs(diff))
                # Filter out near-zero reference values for relative difference
                mask = np.abs(ref_spectral) > 0.01 * np.max(np.abs(ref_spectral))
                if np.any(mask):
                    max_rel = np.max(rel_diff[mask])
                else:
                    max_rel = np.nan
                
                print(f"\n{name} vs {ref_name}:")
                print(f"  Max absolute difference: {max_diff:.6e}")
                print(f"  Mean absolute difference: {mean_diff:.6e}")
                print(f"  Max relative difference (where S > 1% max): {max_rel:.4%}")
            else:
                print(f"\nCannot compare {name} with {ref_name}: different frequency grids")
                print(f"  {name}: {len(omega)} points from {omega[0]:.3f} to {omega[-1]:.3f}")
                print(f"  {ref_name}: {len(ref_omega)} points from {ref_omega[0]:.3f} to {ref_omega[-1]:.3f}")
        
        ax2.set_xlabel('ω', fontsize=12)
        ax2.set_ylabel('Difference', fontsize=12)
        ax2.set_title(f'Difference from {ref_name}', fontsize=12)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Need at least 2 methods for comparison', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    output_file = TEST_DIR / "spectral_comparison.png"
    plt.savefig(output_file, dpi=150)
    print(f"\nComparison plot saved to: {output_file}")
    plt.close()


def run_full_exact_diagonalization_spectral():
    """
    For small systems, we can compute the exact spectral function using 
    full diagonalization as the ultimate reference.
    
    S(q,ω) = Σ_n |⟨n|O|0⟩|² δ(ω - (E_n - E_0))
    
    where |0⟩ is ground state, |n⟩ are all eigenstates.
    
    Note: For this we need all eigenvectors, but for now we'll just
    use the eigenvalues to compute a density of states as a sanity check.
    """
    print("\n" + "=" * 60)
    print("Computing reference from full spectrum (eigenvalues)")
    print("=" * 60)
    
    # Check if we have full spectrum from ED
    eigenvalues_file = TEST_DIR / "output/eigenvalues.txt"
    if not eigenvalues_file.exists():
        eigenvalues_file = TEST_DIR / "output/eigenvalues.dat"
    
    if not eigenvalues_file.exists():
        print("Full spectrum not available. Looking for eigenvalues file...")
        # Try to find it
        for f in (TEST_DIR / "output").rglob("eigenvalues*"):
            print(f"  Found: {f}")
            eigenvalues_file = f
            break
    
    if not eigenvalues_file.exists():
        print("No eigenvalues file found. Run ground state diagonalization first.")
        return None
    
    # Load eigenvalues
    try:
        eigenvalues = np.loadtxt(eigenvalues_file)
        if eigenvalues.ndim == 0:
            eigenvalues = np.array([float(eigenvalues)])
    except Exception as e:
        print(f"Error loading eigenvalues: {e}")
        return None
        
    print(f"Loaded {len(eigenvalues)} eigenvalues from {eigenvalues_file}")
    print(f"Ground state energy: {eigenvalues[0]:.10f}")
    if len(eigenvalues) > 1:
        print(f"First gap: {eigenvalues[1] - eigenvalues[0]:.10f}")
        print(f"Bandwidth: {eigenvalues[-1] - eigenvalues[0]:.6f}")
    
    # Create output directory
    output_dir = TEST_DIR / "exact_reference"
    output_dir.mkdir(exist_ok=True)
    
    # Create frequency grid
    omega_min, omega_max = -5.0, 5.0
    num_omega = 500
    broadening = 0.1
    omega = np.linspace(omega_min, omega_max, num_omega)
    
    # Shifted eigenvalues (excitation energies)
    E0 = eigenvalues[0]
    excitations = eigenvalues - E0
    
    # For a rough approximation of S(q,ω), we compute a weighted density of states
    # where we assume uniform matrix elements |⟨n|O|0⟩|² ≈ const
    # This is NOT the correct spectral function but gives the right peak positions
    
    dos = np.zeros_like(omega)
    for En in excitations:
        dos += broadening / (np.pi * ((omega - En)**2 + broadening**2))
    
    # Normalize by number of states (gives density of states)
    dos /= len(eigenvalues)
    
    # Save
    output_file = output_dir / "density_of_states.txt"
    np.savetxt(output_file, np.column_stack([omega, dos]), 
               header="omega DOS (normalized Lorentzian-broadened)")
    print(f"Saved density of states to: {output_file}")
    
    # Also save eigenvalue statistics
    stats_file = output_dir / "eigenvalue_stats.txt"
    with open(stats_file, 'w') as f:
        f.write(f"# Eigenvalue statistics for 8-site pyrochlore\n")
        f.write(f"# Total states: {len(eigenvalues)}\n")
        f.write(f"# Ground state energy: {E0:.10f}\n")
        if len(eigenvalues) > 1:
            f.write(f"# First excited: {eigenvalues[1]:.10f}\n")
            f.write(f"# First gap: {eigenvalues[1] - E0:.10f}\n")
            f.write(f"# Bandwidth: {eigenvalues[-1] - E0:.6f}\n")
        f.write(f"\n# All excitation energies:\n")
        for i, E in enumerate(excitations):
            f.write(f"{i} {E:.10f}\n")
    print(f"Saved eigenvalue statistics to: {stats_file}")
    
    return output_dir


def main():
    """Main test driver"""
    print("\n" + "=" * 70)
    print("SPECTRAL FUNCTION CONSISTENCY TEST")
    print("Testing 3 methods on 8-site pyrochlore cluster")
    print("=" * 70)
    
    # Create test directory
    TEST_DIR.mkdir(exist_ok=True)
    
    # Step 1: Setup test cluster
    if not setup_test_cluster():
        print("Failed to setup test cluster")
        return 1
    
    # Step 2: Run ground state diagonalization (needed for comparison)
    if not run_ground_state_diagonalization():
        print("Failed to run ground state diagonalization")
        # Continue anyway - some methods don't need pre-computed eigenstates
    
    # Step 3: Run all three spectral methods
    run_ftlm_spectral()
    run_ground_state_dssf_continued_fraction()
    run_spectral_from_state()
    
    # Step 4: Compute exact reference (if possible)
    run_full_exact_diagonalization_spectral()
    
    # Step 5: Compare results
    compare_results()
    
    print("\n" + "=" * 70)
    print("SPECTRAL CONSISTENCY TEST COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
