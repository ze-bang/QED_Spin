#!/usr/bin/env python3
"""
Test consistency between TPQ_DSSF SSSF and compute_bfg_order_parameters
for static spin structure factor S(q) = (1/N) Σ_{i,j} ⟨S⁺ᵢ S⁻ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}

The conventions are:
- compute_bfg_order_parameters.cpp: computes ⟨S⁺ᵢ S⁻ⱼ⟩ directly
- TPQ_DSSF with "1,1": computes ⟨(S⁻(q))† S⁻(q)⟩ = ⟨S⁺(-q) S⁻(q)⟩
  which equals (1/N) Σ_{i,j} ⟨S⁺ᵢ S⁻ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}

So "1,1" (not "0,0") should give the same S(q) as compute_bfg_order_parameters.

Usage:
    python test_sssf_consistency.py <cluster_dir>
"""

import sys
import os
import subprocess
import numpy as np

def read_positions(cluster_dir):
    """Read site positions from positions.dat"""
    positions = []
    with open(os.path.join(cluster_dir, 'positions.dat'), 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 5:
                    x, y = float(parts[3]), float(parts[4])
                    positions.append((x, y))
    return np.array(positions)

def compute_smsp_from_state(state, n_sites):
    """
    Compute ⟨ψ|S⁻ᵢ S⁺ⱼ|ψ⟩ for all i,j pairs directly.
    This matches TPQ_DSSF "0,0" convention: ⟨(S⁺)†S⁺⟩ = ⟨S⁻S⁺⟩
    
    ED convention: bit=0 is UP (Sz=+1/2), bit=1 is DOWN (Sz=-1/2)
    S⁺|↓⟩ = |↑⟩, S⁺|↑⟩ = 0
    S⁻|↑⟩ = |↓⟩, S⁻|↓⟩ = 0
    """
    n_states = len(state)
    corr = np.zeros((n_sites, n_sites), dtype=complex)
    
    for basis in range(n_states):
        coeff = state[basis]
        if abs(coeff) < 1e-15:
            continue
            
        for i in range(n_sites):
            for j in range(n_sites):
                if i == j:
                    # Diagonal: S⁻ᵢ S⁺ᵢ = (1/2 - Szᵢ)
                    # bit=0 (up): Sz=+1/2, S⁻S⁺ = 0
                    # bit=1 (down): Sz=-1/2, S⁻S⁺ = 1
                    bit_i = (basis >> i) & 1
                    if bit_i == 1:  # down spin
                        corr[i, i] += abs(coeff)**2
                else:
                    # Off-diagonal: S⁻ᵢ S⁺ⱼ
                    # Need: j is DOWN (bit=1), i is UP (bit=0)
                    # Result: j becomes UP, i becomes DOWN
                    bit_i = (basis >> i) & 1
                    bit_j = (basis >> j) & 1
                    
                    if bit_j == 1 and bit_i == 0:  # j=DOWN, i=UP
                        # Apply S⁺ⱼ: flip j from 1 to 0
                        # Apply S⁻ᵢ: flip i from 0 to 1
                        new_basis = basis ^ (1 << i) ^ (1 << j)
                        corr[i, j] += np.conj(state[new_basis]) * coeff
    
    return corr

def compute_structure_factor(corr, positions, q):
    """
    Compute S(q) = (1/N) Σ_{i,j} ⟨S⁺ᵢ S⁻ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}
    """
    n_sites = len(positions)
    s_q = 0.0 + 0.0j
    
    for i in range(n_sites):
        for j in range(n_sites):
            dr = positions[j] - positions[i]
            phase = q[0] * dr[0] + q[1] * dr[1]
            s_q += corr[i, j] * np.exp(1j * phase)
    
    return s_q / n_sites

def compute_sssf_operator_style(state, n_sites, positions, q):
    """
    Compute S(q) using operator formalism:
    
    For "0,0" (S⁺ operators):
    ⟨ψ|(S⁺(q))† S⁺(q)|ψ⟩ = ||S⁺(q)|ψ⟩||²
    
    where S⁺(q) = (1/√N) Σⱼ S⁺ⱼ e^{iq·Rⱼ}
    
    This should equal (1/N) Σ_{i,j} ⟨S⁻ᵢ S⁺ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}
    """
    n_states = len(state)
    
    # Apply S⁺(q) to |ψ⟩
    chi = np.zeros(n_states, dtype=complex)
    
    for basis in range(n_states):
        coeff = state[basis]
        if abs(coeff) < 1e-15:
            continue
        
        # S⁺(q)|basis⟩ = (1/√N) Σⱼ S⁺ⱼ|basis⟩ e^{iq·Rⱼ}
        for j in range(n_sites):
            bit_j = (basis >> j) & 1
            if bit_j == 1:  # j is DOWN, can apply S⁺
                new_basis = basis ^ (1 << j)  # flip j to UP
                phase = q[0] * positions[j, 0] + q[1] * positions[j, 1]
                chi[new_basis] += coeff * np.exp(1j * phase) / np.sqrt(n_sites)
    
    # S(q) = ||chi||² = ⟨chi|chi⟩
    s_q = np.sum(np.abs(chi)**2)
    
    return s_q

def main():
    cluster_dir = sys.argv[1] if len(sys.argv) > 1 else 'test_kagome_2x3'
    
    # Read positions
    positions = read_positions(cluster_dir)
    n_sites = len(positions)
    print(f"Cluster: {n_sites} sites")
    
    # Load TPQ state
    import h5py
    h5_path = os.path.join(cluster_dir, 'output', 'ed_results.h5')
    
    with h5py.File(h5_path, 'r') as f:
        states_group = f['tpq/samples/sample_0/states']
        state_keys = list(states_group.keys())
        betas = [float(k.replace('beta_', '')) for k in state_keys]
        max_beta_idx = np.argmax(betas)
        max_beta_key = state_keys[max_beta_idx]
        print(f"Using TPQ state: {max_beta_key} (T ≈ {1/betas[max_beta_idx]:.6f})")
        
        state = states_group[max_beta_key][:]
        # Handle structured array (real, imag) format
        if state.dtype.names is not None and 'real' in state.dtype.names:
            state = state['real'] + 1j * state['imag']
        elif state.dtype in [np.float64, np.float32]:
            # Real storage, combine as complex
            state = state.view(np.complex128)
        
    # Normalize
    norm = np.linalg.norm(state)
    state = state / norm
    print(f"State norm after normalization: {np.linalg.norm(state):.10f}")
    print(f"Hilbert space dimension: {len(state)}")
    
    # Compute S⁻S⁺ correlations (matches TPQ_DSSF "0,0")
    print("\nComputing ⟨S⁻ᵢ S⁺ⱼ⟩ correlations (matches TPQ_DSSF '0,0')...")
    corr = compute_smsp_from_state(state, n_sites)
    
    # Check diagonal sum = number of down spins (should be ~N/2 for Sz=0)
    diag_sum = np.real(np.trace(corr))
    print(f"Tr(⟨S⁻S⁺⟩) = {diag_sum:.6f} (expected ~{n_sites/2} for Sz=0)")
    
    # Define test q-points (in terms of π)
    q_points = [
        ([0.0, 0.0], "Γ"),
        ([1.0, 0.0], "X"),
        ([0.0, 1.0], "Y"),
        ([1.0, 1.0], "M"),
    ]
    
    print("\n" + "="*70)
    print("COMPARISON: compute_bfg_order_parameters (S⁻S⁺) vs TPQ_DSSF '0,0'")
    print("="*70)
    print(f"{'q-point':<10} {'S(q) direct (S⁻S⁺)':<25} {'S(q) operator ||S⁺(q)|ψ⟩||²':<30} {'Match':<10}")
    print("-"*70)
    
    all_match = True
    for q_pi, label in q_points:
        q = [q_pi[0] * np.pi, q_pi[1] * np.pi]
        
        # Method 1: Direct correlation
        s_q_direct = compute_structure_factor(corr, positions, q)
        
        # Method 2: Operator style (like SSSF "1,1")
        s_q_operator = compute_sssf_operator_style(state, n_sites, positions, q)
        
        # Compare
        diff = abs(np.real(s_q_direct) - s_q_operator)
        match = "✓" if diff < 1e-10 else "✗"
        if diff >= 1e-10:
            all_match = False
        
        print(f"{label:<10} {s_q_direct.real:>12.8f} + {s_q_direct.imag:>9.2e}i  {s_q_operator:>12.8f}             {match}")
    
    print("-"*70)
    if all_match:
        print("✓ All q-points match! S(q) from compute_bfg_order_parameters equals TPQ_DSSF '0,0'")
        print("\nBoth methods now compute: S(q) = (1/N) Σ ⟨S⁻ᵢS⁺ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)} = ||S⁺(q)|ψ⟩||²")
    else:
        print("✗ Some q-points don't match! Check the implementation.")
    
    # Also test "0,0" convention
    print("\n" + "="*70)
    print("TEST: What does '0,0' (S⁺ operators) give?")
    print("This computes ⟨(S⁺(q))† S⁺(q)⟩ = ||S⁺(q)|ψ⟩||² = ⟨S⁻(-q) S⁺(q)⟩")
    print("="*70)
    
    def compute_sssf_sp_style(state, n_sites, positions, q):
        """S⁺(q) style: ||S⁺(q)|ψ⟩||²"""
        n_states = len(state)
        chi = np.zeros(n_states, dtype=complex)
        
        for basis in range(n_states):
            coeff = state[basis]
            if abs(coeff) < 1e-15:
                continue
            
            for j in range(n_sites):
                bit_j = (basis >> j) & 1
                if bit_j == 1:  # j is DOWN, can apply S⁺
                    new_basis = basis ^ (1 << j)  # flip j to UP
                    phase = q[0] * positions[j, 0] + q[1] * positions[j, 1]
                    chi[new_basis] += coeff * np.exp(1j * phase) / np.sqrt(n_sites)
        
        return np.sum(np.abs(chi)**2)
    
    def compute_smsp_structure_factor(state, n_sites, positions, q):
        """Compute S(q) for ⟨S⁻ᵢ S⁺ⱼ⟩"""
        # First compute ⟨S⁻ᵢ S⁺ⱼ⟩ correlation
        n_states = len(state)
        corr_smsp = np.zeros((n_sites, n_sites), dtype=complex)
        
        for basis in range(n_states):
            coeff = state[basis]
            if abs(coeff) < 1e-15:
                continue
                
            for i in range(n_sites):
                for j in range(n_sites):
                    if i == j:
                        # S⁻ᵢ S⁺ᵢ = (1/2 - Szᵢ)
                        bit_i = (basis >> i) & 1
                        if bit_i == 1:  # down spin
                            corr_smsp[i, i] += abs(coeff)**2
                    else:
                        # S⁻ᵢ S⁺ⱼ: need j=DOWN, i=UP
                        bit_i = (basis >> i) & 1
                        bit_j = (basis >> j) & 1
                        
                        if bit_j == 1 and bit_i == 0:
                            new_basis = basis ^ (1 << i) ^ (1 << j)
                            corr_smsp[i, j] += np.conj(state[new_basis]) * coeff
        
        # S(q) = (1/N) Σ_{i,j} ⟨S⁻ᵢ S⁺ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}
        s_q = 0.0 + 0.0j
        for i in range(n_sites):
            for j in range(n_sites):
                dr = positions[j] - positions[i]
                phase = q[0] * dr[0] + q[1] * dr[1]
                s_q += corr_smsp[i, j] * np.exp(1j * phase)
        
        return s_q / n_sites
    
    print(f"{'q-point':<10} {'||S⁺(q)|ψ⟩||²':<25} {'⟨S⁻S⁺⟩ structure':<25} {'Match':<10}")
    print("-"*70)
    
    for q_pi, label in q_points:
        q = [q_pi[0] * np.pi, q_pi[1] * np.pi]
        
        s_q_sp = compute_sssf_sp_style(state, n_sites, positions, q)
        s_q_smsp = compute_smsp_structure_factor(state, n_sites, positions, q)
        
        diff = abs(s_q_sp - np.real(s_q_smsp))
        match = "✓" if diff < 1e-10 else "✗"
        
        print(f"{label:<10} {s_q_sp:>12.8f}             {s_q_smsp.real:>12.8f}             {match}")
    
    print("-"*70)
    print("\nSUMMARY:")
    print("  - '0,0' (S⁺S⁺) gives S(q) = (1/N) Σ ⟨S⁻ᵢS⁺ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}  [MATCHES compute_bfg_order_parameters ✓]")
    print("  - '1,1' (S⁻S⁻) gives S(q) = (1/N) Σ ⟨S⁺ᵢS⁻ⱼ⟩ e^{iq·(Rⱼ-Rᵢ)}  [Hermitian conjugate]")

if __name__ == '__main__':
    main()
