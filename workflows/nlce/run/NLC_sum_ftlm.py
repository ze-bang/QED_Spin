#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation for FTLM results.

This script performs NLCE summation using thermodynamic data obtained from
Finite Temperature Lanczos Method (FTLM) calculations on each cluster.

Key differences from standard NLC_sum.py:
- Reads FTLM output files (ed_results.h5 or legacy ftlm_thermo.txt) instead of eigenvalue files
- Handles error propagation from FTLM sampling uncertainties
- Works with pre-computed thermodynamic quantities
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
from collections import defaultdict

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. HDF5 file reading will not be available.")


class NLCExpansionFTLM:
    def __init__(self, cluster_dir, ftlm_dir, temp_min, temp_max, num_temps, SI_units=False):
        """
        Initialize the NLC expansion calculator for FTLM data.
        
        Args:
            cluster_dir: Directory containing cluster information files
            ftlm_dir: Directory containing FTLM output files
            temp_min: Minimum temperature
            temp_max: Maximum temperature  
            num_temps: Number of temperature points
            SI_units: Flag for SI units
        """
        self.cluster_dir = cluster_dir
        self.ftlm_dir = ftlm_dir
        self.SI = SI_units
        
        # Generate temperature grid matching FTLM calculations
        self.temp_values = np.logspace(
            np.log(temp_min)/np.log(10), 
            np.log(temp_max)/np.log(10), 
            num_temps
        )
        
        self.clusters = {}  # {cluster_id: {order, multiplicity, thermo_data, ...}}
        self.weights = {}   # Calculated weights for each cluster
        self.subcluster_info = {}
        
    def read_clusters(self):
        """Read all cluster information from files."""
        pattern = os.path.join(self.cluster_dir, "cluster_*_order_*.dat")
        cluster_files = glob.glob(pattern)
        
        print(f"Found {len(cluster_files)} cluster info files")
        
        for file_path in cluster_files:
            basename = os.path.basename(file_path)
            match = re.search(r'cluster_(\d+)_order_(\d+)', basename)
            if not match:
                continue
                
            cluster_id = int(match.group(1))
            order = int(match.group(2))
            
            # Read multiplicity from file
            with open(file_path, 'r') as f:
                multiplicity = None
                for line in f:
                    if line.startswith("# Multiplicity") and ":" in line:
                        # Handle both "# Multiplicity:" and "# Multiplicity (lattice constant):"
                        multiplicity = float(line.split(":")[-1].strip())
                        break
                
                if multiplicity is None:
                    print(f"Warning: Could not find multiplicity for cluster {cluster_id}, using default 1.0")
                    multiplicity = 1.0
            
            self.clusters[cluster_id] = {
                'order': order,
                'multiplicity': multiplicity,
                'file_path': file_path
            }
        
        print(f"Loaded {len(self.clusters)} clusters")
    
    def _read_ftlm_from_hdf5(self, h5_file, cluster_id):
        """Read FTLM data from HDF5 file."""
        try:
            with h5py.File(h5_file, 'r') as f:
                # Check for FTLM averaged data
                if '/ftlm/averaged' not in f:
                    return None
                
                ftlm_grp = f['/ftlm/averaged']
                
                # Check required datasets exist
                required = ['temperatures', 'energy', 'specific_heat', 'entropy', 'free_energy']
                if not all(key in ftlm_grp for key in required):
                    return None
                
                temps = ftlm_grp['temperatures'][:]
                energy = ftlm_grp['energy'][:]
                spec_heat = ftlm_grp['specific_heat'][:]
                entropy = ftlm_grp['entropy'][:]
                free_energy = ftlm_grp['free_energy'][:]
                
                # Get error bars if available
                energy_err = ftlm_grp['energy_error'][:] if 'energy_error' in ftlm_grp else np.zeros_like(energy)
                spec_heat_err = ftlm_grp['specific_heat_error'][:] if 'specific_heat_error' in ftlm_grp else np.zeros_like(spec_heat)
                entropy_err = ftlm_grp['entropy_error'][:] if 'entropy_error' in ftlm_grp else np.zeros_like(entropy)
                free_energy_err = ftlm_grp['free_energy_error'][:] if 'free_energy_error' in ftlm_grp else np.zeros_like(free_energy)
                
                return {
                    'temperatures': temps,
                    'energy': energy,
                    'energy_error': energy_err,
                    'specific_heat': spec_heat,
                    'specific_heat_error': spec_heat_err,
                    'entropy': entropy,
                    'entropy_error': entropy_err,
                    'free_energy': free_energy,
                    'free_energy_error': free_energy_err
                }
        except Exception as e:
            print(f"Warning: Error reading HDF5 FTLM data for cluster {cluster_id}: {e}")
            return None
    
    def _read_ftlm_from_txt(self, ftlm_file, cluster_id):
        """Read FTLM data from legacy text file."""
        try:
            data = np.loadtxt(ftlm_file)
            return {
                'temperatures': data[:, 0],
                'energy': data[:, 1],
                'energy_error': data[:, 2],
                'specific_heat': data[:, 3],
                'specific_heat_error': data[:, 4],
                'entropy': data[:, 5],
                'entropy_error': data[:, 6],
                'free_energy': data[:, 7],
                'free_energy_error': data[:, 8]
            }
        except Exception as e:
            print(f"Error reading FTLM text file for cluster {cluster_id}: {e}")
            return None
    
    def read_ftlm_data(self):
        """Read FTLM thermodynamic data for each cluster (HDF5 or legacy text format)."""
        print("\nReading FTLM data...")
        
        for cluster_id in self.clusters:
            order = self.clusters[cluster_id]['order']
            cluster_output_dir = os.path.join(
                self.ftlm_dir, 
                f'cluster_{cluster_id}_order_{order}',
                'output'
            )
            
            data = None
            
            # Try HDF5 file first (new format) - check both possible locations
            h5_file = os.path.join(cluster_output_dir, "thermo", "ed_results.h5")
            if not os.path.exists(h5_file):
                h5_file = os.path.join(cluster_output_dir, "ed_results.h5")
            if HAS_H5PY and os.path.exists(h5_file):
                data = self._read_ftlm_from_hdf5(h5_file, cluster_id)
            
            # Fall back to legacy text file format
            if data is None:
                ftlm_file = os.path.join(cluster_output_dir, 'thermo', 'ftlm_thermo.txt')
                if os.path.exists(ftlm_file):
                    data = self._read_ftlm_from_txt(ftlm_file, cluster_id)
            
            if data is None:
                print(f"Warning: FTLM data not found for cluster {cluster_id}")
                self.clusters[cluster_id]['has_data'] = False
                continue
            
            # Interpolate if temperature grid doesn't match
            temps = data['temperatures']
            if len(temps) != len(self.temp_values) or not np.allclose(temps, self.temp_values, rtol=1e-3):
                print(f"  Cluster {cluster_id}: interpolating from {len(temps)} to {len(self.temp_values)} temperature points")
                from scipy.interpolate import interp1d
                
                for key in ['energy', 'energy_error', 'specific_heat', 'specific_heat_error',
                           'entropy', 'entropy_error', 'free_energy', 'free_energy_error']:
                    interp_func = interp1d(temps, data[key], kind='cubic', fill_value='extrapolate')
                    data[key] = interp_func(self.temp_values)
            
            self.clusters[cluster_id]['thermo_data'] = {
                'energy': data['energy'],
                'energy_error': data['energy_error'],
                'specific_heat': data['specific_heat'],
                'specific_heat_error': data['specific_heat_error'],
                'entropy': data['entropy'],
                'entropy_error': data['entropy_error'],
                'free_energy': data['free_energy'],
                'free_energy_error': data['free_energy_error']
            }
            self.clusters[cluster_id]['has_data'] = True
            
            print(f"  Cluster {cluster_id} (order {order}): loaded {len(self.temp_values)} temperature points")
    
    def read_subcluster_info(self):
        """Read subcluster information from file."""
        subcluster_file = os.path.join(self.cluster_dir, 'subclusters_info.txt')
        
        if not os.path.exists(subcluster_file):
            print(f"Warning: Subcluster info file not found: {subcluster_file}")
            print("Using simple order-based heuristic for subclusters")
            return
        
        print(f"Reading subcluster information from: {subcluster_file}")
        
        with open(subcluster_file, 'r') as f:
            current_cluster_id = None
            
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse cluster header: "Cluster X (Order Y):"
                if line.startswith('Cluster') and '(Order' in line:
                    # Extract cluster ID from "Cluster X (Order Y):"
                    try:
                        cluster_part = line.split('(')[0].strip()  # "Cluster X"
                        current_cluster_id = int(cluster_part.split()[-1])  # Extract X
                        self.subcluster_info[current_cluster_id] = {}
                    except (ValueError, IndexError):
                        print(f"Warning: Could not parse cluster ID from: {line}")
                        current_cluster_id = None
                    continue
                
                # Parse subcluster info
                if current_cluster_id is not None:
                    if "No subclusters" in line:
                        self.subcluster_info[current_cluster_id] = {}
                    elif "Subclusters:" in line:
                        # Extract subcluster info: "Subclusters: (1, 2)"
                        subcluster_str = line.split('Subclusters:')[1].strip()
                        subcluster_dict = {}
                        
                        # Parse tuples: (id, mult)
                        # Handle both single tuple "(1, 2)" and list "[(1, 2), (3, 4)]"
                        subcluster_str = subcluster_str.strip('[]')
                        
                        # Split by '),' to separate multiple tuples
                        for item in subcluster_str.split('),'):
                            item = item.strip().strip('(').strip(')')
                            if item:
                                parts = item.split(',')
                                if len(parts) == 2:
                                    try:
                                        sub_id = int(parts[0].strip())
                                        mult = int(parts[1].strip())
                                        subcluster_dict[sub_id] = mult
                                    except ValueError:
                                        print(f"Warning: Could not parse subcluster: {item}")
                        
                        self.subcluster_info[current_cluster_id] = subcluster_dict
        
        print(f"Loaded subcluster info for {len(self.subcluster_info)} clusters")
    
    def get_subclusters(self, cluster_id):
        """Get all subclusters of a given cluster with their multiplicities."""
        if hasattr(self, 'subcluster_info') and cluster_id in self.subcluster_info:
            return self.subcluster_info[cluster_id]
        
        # Fallback: use simple order-based heuristic
        subclusters = {}
        order = self.clusters[cluster_id]['order']
        for cid, data in self.clusters.items():
            if data['order'] < order:
                subclusters[cid] = 1  # Assume multiplicity 1
        
        return subclusters
    
    def calculate_weights(self, verbose=True):
        """
        Calculate weights for all clusters using the NLC principle.
        
        Weight formula: W(c) = P(c) - Σ_s Y_{c,s} * W(s)
        where:
          - P(c) = property of cluster c (e.g., energy from ED/FTLM)
          - Y_{c,s} = multiplicity of subcluster s in cluster c
          - W(s) = weight of subcluster s (calculated recursively)
        
        Args:
            verbose: If True, print detailed weight calculation breakdown
        """
        self.read_subcluster_info()
        
        # Sort clusters by order
        sorted_clusters = sorted(
            [(cid, data['order']) for cid, data in self.clusters.items()],
            key=lambda x: x[1]
        )
        
        # Initialize weights and weight errors for each thermodynamic quantity
        self.weights = {
            'energy': {},
            'specific_heat': {},
            'entropy': {},
            'free_energy': {}
        }
        
        self.weight_errors = {
            'energy': {},
            'specific_heat': {},
            'entropy': {},
            'free_energy': {}
        }
        
        print("\n" + "="*100)
        print("CALCULATING NLC WEIGHTS")
        print("="*100)
        print("Formula: W(c) = P(c) - Σ_s Y_{c,s} * W(s)")
        print("  where P(c) = cluster property, Y_{c,s} = subcluster multiplicity, W(s) = subcluster weight")
        print("="*100)
        
        # Pick representative temperature indices for verbose output
        n_temps = len(self.temp_values)
        T_indices = {
            'low': max(0, n_temps // 10),
            'mid': n_temps // 2,
            'high': min(n_temps - 1, 9 * n_temps // 10)
        }
        
        for cluster_id, order in sorted_clusters:
            if not self.clusters[cluster_id].get('has_data', False):
                if verbose:
                    print(f"\n  Cluster {cluster_id} (order {order}): SKIPPED (no data)")
                continue
            
            multiplicity = self.clusters[cluster_id]['multiplicity']
            subclusters = self.get_subclusters(cluster_id)
            
            # Get thermodynamic data for this cluster
            thermo = self.clusters[cluster_id]['thermo_data']
            
            if verbose:
                print(f"\n{'─'*100}")
                print(f"  CLUSTER {cluster_id} (order={order}, L={multiplicity:.4f})")
                print(f"{'─'*100}")
                print(f"    Subclusters: {dict(subclusters) if subclusters else 'None (base cluster)'}")
                
                # Show raw cluster properties P(c) for ALL quantities
                print(f"\n    RAW CLUSTER PROPERTIES P(c) from FTLM:")
                print(f"      {'T':>10s}  {'P(E)':>12s}  {'P(Cv)':>12s}  {'P(S)':>12s}  {'P(F)':>12s}")
                print(f"      {'-'*66}")
                for label, idx in T_indices.items():
                    T = self.temp_values[idx]
                    P_E = thermo['energy'][idx]
                    P_Cv = thermo['specific_heat'][idx]
                    P_S = thermo['entropy'][idx]
                    P_F = thermo['free_energy'][idx]
                    print(f"      {T:10.4f}  {P_E:+12.6f}  {P_Cv:+12.6f}  {P_S:+12.6f}  {P_F:+12.6f}")
            
            # For each quantity, weight = cluster_value - sum of (subcluster_multiplicity * subcluster_weight)
            # NOTE: We do NOT multiply by cluster multiplicity here - that happens in sum_nlc()
            # Error propagation: σ²_weight = σ²_cluster + Σ(sub_mult² * σ²_subcluster_weight)
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                cluster_value = thermo[quantity]
                cluster_error = thermo[f'{quantity}_error']
                
                weight_value = cluster_value.copy()
                weight_error_sq = cluster_error**2
                
                subcluster_contribution = np.zeros_like(cluster_value)
                
                # Subtract weighted contributions from subclusters
                for sub_id, sub_mult in subclusters.items():
                    if sub_id in self.weights[quantity]:
                        sub_contrib = sub_mult * self.weights[quantity][sub_id]
                        weight_value -= sub_contrib
                        subcluster_contribution += sub_contrib
                        # Add subcluster weight error in quadrature
                        weight_error_sq += (sub_mult * self.weight_errors[quantity][sub_id])**2
                
                self.weights[quantity][cluster_id] = weight_value
                self.weight_errors[quantity][cluster_id] = np.sqrt(weight_error_sq)
            
            # Verbose output for ALL quantities (not just energy)
            if verbose:
                print(f"\n    WEIGHT CALCULATIONS W(c) = P(c) - Σ Y_cs × W(s):")
                print(f"      {'T':>10s}  {'W(E)':>12s}  {'W(Cv)':>12s}  {'W(S)':>12s}  {'W(F)':>12s}")
                print(f"      {'-'*66}")
                for label, idx in T_indices.items():
                    T = self.temp_values[idx]
                    W_E = self.weights['energy'][cluster_id][idx]
                    W_Cv = self.weights['specific_heat'][cluster_id][idx]
                    W_S = self.weights['entropy'][cluster_id][idx]
                    W_F = self.weights['free_energy'][cluster_id][idx]
                    print(f"      {T:10.4f}  {W_E:+12.6f}  {W_Cv:+12.6f}  {W_S:+12.6f}  {W_F:+12.6f}")
                
                # Show contribution to final sum: L(c) * W(c)
                print(f"\n    CONTRIBUTIONS L(c) × W(c) = {multiplicity:.4f} × W(c):")
                print(f"      {'T':>10s}  {'L×W(E)':>12s}  {'L×W(Cv)':>12s}  {'L×W(S)':>12s}  {'L×W(F)':>12s}")
                print(f"      {'-'*66}")
                for label, idx in T_indices.items():
                    T = self.temp_values[idx]
                    LW_E = multiplicity * self.weights['energy'][cluster_id][idx]
                    LW_Cv = multiplicity * self.weights['specific_heat'][cluster_id][idx]
                    LW_S = multiplicity * self.weights['entropy'][cluster_id][idx]
                    LW_F = multiplicity * self.weights['free_energy'][cluster_id][idx]
                    print(f"      {T:10.4f}  {LW_E:+12.6f}  {LW_Cv:+12.6f}  {LW_S:+12.6f}  {LW_F:+12.6f}")
    
    def euler_resummation(self, partial_sums, l=3):
        """
        Apply Euler transformation for alternating series.
        
        Implementation follows Tang–Khatami–Rigol (arXiv:1207.3366):
        Keep first l bare terms, apply Euler transform to the tail using
        forward differences with 2^{-(k+1)} weights.
        
        Args:
            partial_sums: Array of partial sums S_n at each order
            l: Number of highest-order terms to keep "as is" (default: 3)
            
        Returns:
            Tuple of (accelerated_sum, error_estimate)
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else np.zeros_like(self.temp_values), np.zeros_like(self.temp_values)
        
        # Convert partial sums to increments (the a_n terms)
        increments = [partial_sums[0]]
        for i in range(1, n):
            increments.append(partial_sums[i] - partial_sums[i-1])
        
        # Choose l adaptively if sequence is short
        l_use = min(l, max(2, n - 3))
        
        if n <= l_use:
            # Not enough terms for Euler, return last partial sum
            return partial_sums[-1], np.abs(increments[-1]) if len(increments) > 0 else np.zeros_like(self.temp_values)
        
        # Keep first (n - l_use) bare terms as-is
        bare_sum = partial_sums[n - l_use - 1] if n - l_use - 1 >= 0 else np.zeros_like(self.temp_values)
        
        # Apply Euler transform to the last l_use increments
        tail_increments = increments[n - l_use:]
        
        # Build forward difference triangle
        diff_triangle = [tail_increments]
        for k in range(len(tail_increments) - 1):
            prev_row = diff_triangle[-1]
            next_row = [prev_row[i+1] - prev_row[i] for i in range(len(prev_row) - 1)]
            if len(next_row) == 0:
                break
            diff_triangle.append(next_row)
        
        # Accumulate Euler tail: sum_k Δ^k a_{n-l}/2^{k+1}
        euler_tail = np.zeros_like(self.temp_values)
        for k, diff_row in enumerate(diff_triangle):
            if len(diff_row) > 0:
                euler_tail += diff_row[0] / (2**(k+1))
        
        result = bare_sum + euler_tail
        
        # Error estimate: use last difference term magnitude
        if len(diff_triangle) > 1 and len(diff_triangle[-1]) > 0:
            error_est = np.abs(diff_triangle[-1][0]) / (2**len(diff_triangle))
        else:
            error_est = np.abs(tail_increments[-1]) / (2**len(tail_increments))
        
        return result, error_est
    
    def wynn_epsilon(self, sequence, return_all_evens=False):
        """
        Wynn's epsilon algorithm for series acceleration.
        
        Standard implementation following Wynn (1956) and standard references.
        Recurrence: ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1/(ε_k^{(n+1)} - ε_k^{(n)})
        
        CRITICAL: Uses GLOBAL abort on instability - if Wynn fails at ANY temperature,
        returns direct sum for ALL temperatures to avoid discontinuities.
        
        Args:
            sequence: List of partial sums [S_0, S_1, ..., S_{N-1}]
            return_all_evens: If True, return list of all even ε values
            
        Returns:
            If return_all_evens: list of [ε_2, ε_4, ...] 
            Otherwise: (best_estimate, error_estimate)
        """
        n = len(sequence)
        
        # Need at least 5 terms for meaningful acceleration
        if n < 5:
            result = sequence[-1]
            error = np.abs(sequence[-1] - sequence[-2]) if n > 1 else np.zeros_like(result)
            return [result] if return_all_evens else (result, error)
        
        seq_arr = np.array(sequence)
        n_temps = len(self.temp_values)
        
        # Determine reasonable scale for blow-up detection
        # Use max absolute value of partial sums, with safety factor
        seq_scale = np.max(np.abs(seq_arr), axis=0) + 1e-10
        max_reasonable = 100.0 * seq_scale  # Allow 100x the typical scale before aborting
        
        # Initialize ε table
        # Column k contains ε_k^{(0)}, ε_k^{(1)}, ..., ε_k^{(n-k-1)}
        # Start with ε_{-1} = 0 and ε_0 = S_n
        eps_table = []
        eps_table.append(np.zeros((n, n_temps)))      # ε_{-1}^{(n)} = 0 for all n
        eps_table.append(seq_arr.copy())               # ε_0^{(n)} = S_n
        
        evens = []
        
        # Build ε table column by column (k = 1, 2, 3, ...)
        for k in range(1, n):
            prev_col = eps_table[k-1]  # ε_{k-1}, shape (n-k+1, n_temps)
            curr_col = eps_table[k]    # ε_k, shape (n-k, n_temps)
            
            n_entries = curr_col.shape[0] - 1  # Number of rows minus 1
            if n_entries <= 0:
                break
            
            next_col = np.zeros((n_entries, n_temps))
            
            # Compute ε_{k+1}^{(i)} for i = 0, ..., n_entries-1
            for i in range(n_entries):
                # ε_{k+1}^{(i)} = ε_{k-1}^{(i+1)} + 1/(ε_k^{(i+1)} - ε_k^{(i)})
                denom = curr_col[i+1] - curr_col[i]
                
                # Check for small denominators (indicates convergence or instability)
                small_mask = np.abs(denom) < 1e-14
                if np.all(small_mask):
                    # All temperatures have zero denominator - algorithm has converged
                    # Stop here and use the last even entry
                    break
                elif np.any(small_mask):
                    # Some temperatures have issues - GLOBAL abort
                    result = sequence[-1]
                    error = np.abs(sequence[-1] - sequence[-2])
                    return [result] if return_all_evens else (result, error)
                
                # Apply Wynn recursion
                next_col[i] = prev_col[i+1] + 1.0 / denom
                
                # GLOBAL blow-up check: abort if ANY temperature produces unreasonable value
                # Use scale-based threshold: if result is 10x larger than any partial sum, abort
                if np.any(np.abs(next_col[i]) > max_reasonable):
                    # Result blowing up relative to input scale - abort globally
                    result = sequence[-1]
                    error = np.abs(sequence[-1] - sequence[-2])
                    return [result] if return_all_evens else (result, error)
            
            # If we broke out of the loop early, stop building table
            if i < n_entries - 1:
                break
            
            eps_table.append(next_col)
            
            # Store even columns (k=2 → ε_2, k=4 → ε_4, etc.)
            if k % 2 == 0 and len(next_col) > 0:
                evens.append(next_col[0].copy())  # Use ε_k^{(0)} as accelerated value
        
        # Return results
        if return_all_evens:
            return evens if len(evens) > 0 else [sequence[-1]]
        
        # Use highest-order even entry as best estimate
        if len(evens) >= 2:
            best = evens[-1]
            error = np.abs(evens[-1] - evens[-2])
        elif len(evens) == 1:
            best = evens[0]
            error = np.abs(evens[0] - sequence[-1])
        else:
            # No even entries computed - fall back to direct
            best = sequence[-1]
            error = np.abs(sequence[-1] - sequence[-2])
        
        # FINAL SANITY CHECK: If Wynn result is unreasonably large compared to input scale,
        # abort and return direct sum. This catches cases where Wynn produces spikes.
        # Use 5x threshold: if any temperature has |result| > 5 * max(|partial_sums|), abort
        spike_mask = np.abs(best) > 5.0 * seq_scale
        if np.any(spike_mask):
            # Wynn produced spike - abort globally and warn
            n_spikes = np.sum(spike_mask)
            print(f"  ⚠ Wynn unstable at {n_spikes}/{n_temps} temperatures (spike detected)")
            print(f"    → Falling back to direct sum (alternating series? Use --resummation=euler)")
            best = sequence[-1]
            error = np.abs(sequence[-1] - sequence[-2])
        
        return best, error
    
    def brezinski_theta(self, sequence, return_all_evens=False):
        """
        Apply Brezinski's θ-algorithm for series acceleration.
        
        Alternative non-linear accelerator, often competitive with Wynn ε.
        Good for both linear and logarithmic convergence.
        
        Args:
            sequence: Array of partial sums S_n at each order
            return_all_evens: If True, return all even θ entries for stability analysis
            
        Returns:
            If return_all_evens: list of even θ entries
            Otherwise: tuple of (best_estimate, error_estimate)
        """
        n = len(sequence)
        if n < 3:
            result = sequence[-1] if n > 0 else np.zeros_like(self.temp_values)
            if return_all_evens:
                return [result]
            return result, np.zeros_like(self.temp_values)
        
        eps_den = 1e-14
        
        # Maximum allowed value to prevent blow-up (similar to Wynn)
        seq_arr = np.array(sequence)
        seq_range = np.max(np.abs(seq_arr), axis=0) + 1e-10
        max_allowed = 100 * seq_range
        
        # Initialize θ table: θ₋₁⁽ⁿ⁾ = 0, θ₀⁽ⁿ⁾ = Sₙ
        # Standard Brezinski form uses two-step recursion for θ₂ₖ₊₁ and θ₂ₖ₊₂
        theta_prev = np.zeros((n, len(self.temp_values)))  # θ_{k-1}
        theta_curr = np.array(sequence)                     # θ_0 = S_n
        
        evens = []  # θ₀ - start empty, will collect θ₂, θ₄, ...
        
        iteration = 0
        while theta_curr.shape[0] > 1:
            iteration += 1
            n_curr = theta_curr.shape[0]
            theta_next = np.zeros((n_curr - 1, len(self.temp_values)))
            
            for i in range(n_curr - 1):
                # Standard θ recursion: θ_{k+1}^{(n)} = θ_{k-1}^{(n+1)} + 1/(θ_k^{(n+1)} - θ_k^{(n)})
                # This is the same form as Wynn ε, but convergence properties differ
                denom = theta_curr[i+1] - theta_curr[i]
                mask_small = np.abs(denom) < eps_den
                
                raw_result = np.where(
                    mask_small,
                    theta_curr[i+1],
                    theta_prev[i+1] + 1.0 / denom
                )
                
                # Clamp to prevent blow-up
                mask_blowup = np.abs(raw_result) > max_allowed
                theta_next[i] = np.where(
                    mask_blowup,
                    theta_curr[i+1],
                    raw_result
                )
            
            if np.all(np.isnan(theta_next)) or np.all(np.isinf(theta_next)):
                break
            
            theta_prev = theta_curr
            theta_curr = theta_next
            
            # Store even entries on even iterations (θ₂, θ₄, ...)
            if iteration % 2 == 0:
                candidate = theta_curr[0].copy()
                if not np.any(np.abs(candidate) > max_allowed):
                    evens.append(candidate)
        
        if return_all_evens:
            return evens
        
        if len(evens) >= 2:
            best = evens[-1]
            error = np.abs(evens[-1] - evens[-2])
        elif len(evens) == 1:
            best = evens[0]
            error = np.abs(sequence[-1] - evens[0]) if len(sequence) > 0 else np.zeros_like(self.temp_values)
        else:
            best = sequence[-1] if len(sequence) > 0 else np.zeros_like(self.temp_values)
            error = np.abs(sequence[-1] - sequence[-2]) if len(sequence) > 1 else np.zeros_like(self.temp_values)
        
        return best, error

    def analyze_convergence(self, partial_sums):
        """
        Comprehensive convergence analysis for NLCE series.
        
        Detects:
        - Alternating behavior (key for Euler transform)
        - Convergence rate (ratio test)
        - Monotonicity
        - Stability across temperature range
        
        Args:
            partial_sums: List of partial sums by order [S_1, S_2, ..., S_N]
            
        Returns:
            Dictionary with convergence diagnostics
        """
        n = len(partial_sums)
        if n < 2:
            return {
                'converged': False,
                'oscillatory': False,
                'alternating_tail': False,
                'ratio_test': None,
                'monotonic': False,
                'stable': False,
                'n_terms': n
            }
        
        # Convert to array for easier manipulation
        partial_sums = np.array(partial_sums)
        
        # Calculate increments (differences between successive orders)
        increments = np.diff(partial_sums, axis=0)
        
        # 1) Check for alternating behavior (critical for Euler)
        # Look at last ~6 increments for sign flips
        if n >= 7:
            recent_increments = increments[-6:]
        elif n >= 3:
            recent_increments = increments
        else:
            recent_increments = increments
            
        if n >= 3:
            # Check if signs consistently alternate across temperatures
            # We check if > 20% of temperatures show alternating behavior
            signs = np.sign(recent_increments)
            # product of consecutive signs: (n_recent-1, n_temps)
            # If alternating, product should be -1.
            is_alternating_per_temp = np.all(signs[1:] * signs[:-1] < 0, axis=0)
            alternating_tail = np.mean(is_alternating_per_temp) > 0.2
        else:
            alternating_tail = False
        
        # 2) Convergence test (magnitudes decreasing)
        if n > 3:
            # Check if magnitude of increments is decreasing for > 50% of temps
            recent_magnitudes = np.abs(increments[-3:])
            is_converging_per_temp = (recent_magnitudes[-1] <= recent_magnitudes[-2]) & \
                                     (recent_magnitudes[-2] <= recent_magnitudes[-3])
            converged = np.mean(is_converging_per_temp) > 0.5
        else:
            converged = False
        
        # 3) Oscillatory (weaker than alternating)
        if n > 4:
            signs = np.sign(increments[-4:])
            # Count sign flips per temperature
            flips_per_temp = np.sum(signs[1:] * signs[:-1] < 0, axis=0)
            # If significant number of temps have >= 2 flips
            oscillatory = np.mean(flips_per_temp >= 2) > 0.2
        else:
            oscillatory = False
        
        # 4) Ratio test for convergence rate
        if n > 2:
            # |a_{n+1}| / |a_n|
            ratios = np.abs(increments[1:]) / (np.abs(increments[:-1]) + 1e-15)
            # Average ratio over last few terms
            avg_ratio_per_temp = np.mean(ratios[-min(3, len(ratios)):], axis=0)
            # Use the WORST (max) ratio across temperatures to be conservative
            # Take 90th percentile to avoid single point spikes but capture bad regions
            avg_ratio = np.percentile(avg_ratio_per_temp, 90)
        else:
            avg_ratio = None
        
        # 5) Monotonicity (all increments same sign)
        if n > 2:
            # Check if > 80% of temps are monotonic
            signs = np.sign(increments)
            is_monotonic_per_temp = np.all(signs == signs[0:1], axis=0)
            monotonic = np.mean(is_monotonic_per_temp) > 0.8
        else:
            monotonic = False
        
        # 6) Stability: check variance across temperature points
        if n > 2:
            # Standard deviation of ratios across temperatures should be small
            last_ratios = ratios[-1] if n > 2 else np.ones(len(self.temp_values))
            ratio_std = np.std(last_ratios)
            stable = ratio_std < 0.5  # Heuristic threshold
        else:
            stable = False
        
        return {
            'converged': converged,
            'oscillatory': oscillatory,
            'alternating_tail': alternating_tail,  # Key for Euler
            'ratio_test': avg_ratio,
            'monotonic': monotonic,
            'stable': stable,
            'n_terms': n
        }
    
    def select_resummation_method(self, partial_sums, verbose=True):
        """
        Simple, clear resummation method selection.
        
        Logic:
        1. Alternating series → Euler (designed for this case)
        2. Diverging series (ratio > 1) → Direct (no acceleration)
        3. Converging series, enough terms → Wynn (NLCE standard)
        4. Too few terms → Direct
        
        Args:
            partial_sums: List of partial sums [S_1, ..., S_N]
            verbose: Print diagnostic information
            
        Returns:
            String indicating selected method
        """
        conv = self.analyze_convergence(partial_sums)
        n = conv['n_terms']
        
        if verbose:
            ratio_str = f"{conv['ratio_test']:.3f}" if conv['ratio_test'] is not None else 'N/A'
            print(f"\n    ===== CONVERGENCE ANALYSIS =====")
            print(f"    Terms: {n}, Ratio: {ratio_str}, Alternating: {conv['alternating_tail']}")
        
        # Simple decision tree
        
        # 1) Too few terms → use what we have
        if n <= 3:
            if verbose:
                print(f"    → Method: DIRECT (too few terms)")
            return 'direct'
        
        # 2) Alternating → Euler (handles both converging and diverging alternating series)
        if conv['alternating_tail']:
            if verbose:
                print(f"    → Method: EULER (alternating series)")
            return 'euler'
        
        # 3) Diverging (ratio > 1) → no reliable acceleration, use highest order available
        if conv['ratio_test'] is not None and conv['ratio_test'] > 1.0:
            if verbose:
                print(f"    → Method: DIRECT (diverging series, ratio={conv['ratio_test']:.2f} > 1)")
            return 'direct'
        
        # 4) Converging, enough terms → Wynn (standard NLCE accelerator)
        if n >= 5:
            if verbose:
                print(f"    → Method: WYNN (converging series, N≥5)")
            return 'wynn'
        
        # 5) Default: Euler (safest for N=4)
        if verbose:
            print(f"    → Method: EULER (default, N=4)")
        return 'euler'
    
    def apply_resummation(self, partial_sums, method='auto', l_euler=3):
        """
        Apply resummation with robust pipeline and stability checks.
        
        Follows NLCE best practices:
        - Run multiple accelerators when possible
        - Check agreement for confidence
        - Use spread among methods for error bars
        
        Args:
            partial_sums: List of partial sums [S_1, ..., S_N]
            method: 'auto', 'direct', 'euler', 'wynn', 'theta', or 'robust'
            l_euler: Number of terms to keep for Euler transform
            
        Returns:
            Dictionary with:
                'value': best estimate
                'error': error estimate
                'method': method(s) used
                'stability': dict with cross-checks
        """
        if len(partial_sums) == 0:
            return {
                'value': np.zeros_like(self.temp_values),
                'error': np.zeros_like(self.temp_values),
                'method': 'none',
                'stability': {}
            }
        
        n = len(partial_sums)
        
        # Auto-select method
        if method == 'auto':
            method = self.select_resummation_method(partial_sums, verbose=True)
        
        # "Robust" means: run multiple methods and check agreement
        if method == 'robust' and n >= 4:
            results = {}
            
            # Always try Euler (safe for alternating and converging)
            euler_val, euler_err = self.euler_resummation(partial_sums, l=l_euler)
            results['euler'] = (euler_val, euler_err)
            
            # Try Wynn if we have enough terms
            if n >= 5:
                wynn_val, wynn_err = self.wynn_epsilon(partial_sums)
                results['wynn'] = (wynn_val, wynn_err)
            
            # Include direct sum
            direct = partial_sums[-1]
            results['direct'] = (direct, np.abs(partial_sums[-1] - partial_sums[-2]) if n > 1 else np.zeros_like(direct))
            
            # Central value: average of accelerators
            accelerator_values = [v[0] for k, v in results.items() if k != 'direct']
            best_val = np.mean(accelerator_values, axis=0)
            
            # Error: spread among methods + individual errors
            all_values = [v[0] for v in results.values()]
            all_errors = [v[1] for v in results.values()]
            spread = np.std(all_values, axis=0)
            avg_method_error = np.mean(all_errors, axis=0)
            total_error = np.sqrt(spread**2 + avg_method_error**2)
            
            return {
                'value': best_val,
                'error': total_error,
                'method': 'robust: ' + ', '.join(results.keys()),
                'stability': {
                    'all_results': results,
                    'spread': spread,
                    'agreement': spread / (np.abs(best_val) + 1e-15)
                }
            }
        
        # Single method application
        if method == 'direct':
            val = partial_sums[-1]
            err = np.abs(partial_sums[-1] - partial_sums[-2]) if n > 1 else np.zeros_like(val)
        elif method == 'euler':
            val, err = self.euler_resummation(partial_sums, l=l_euler)
        elif method == 'wynn':
            val, err = self.wynn_epsilon(partial_sums)
        else:
            print(f"  Warning: Unknown method '{method}', using direct")
            val = partial_sums[-1]
            err = np.abs(partial_sums[-1] - partial_sums[-2]) if n > 1 else np.zeros_like(val)
        
        return {
            'value': val,
            'error': err,
            'method': method,
            'stability': {}
        }
    
    def derivative_spline(self, x_data, y_data, smoothing=0.01, monotonic=True):
        """
        Compute derivative using shape-constrained spline.
        
        For C(T) = dE/dT, enforce C ≥ 0 by using monotonic constraint on E(T).
        Uses scipy's UnivariateSpline with monotonicity checking.
        
        Args:
            x_data: Temperature values
            y_data: Energy values E(T)
            smoothing: Spline smoothing parameter (s)
            monotonic: Enforce monotonic increasing E(T)
            
        Returns:
            Derivative values (specific heat)
        """
        try:
            from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
        except ImportError:
            print("Warning: scipy not available, using simple finite differences")
            # Fallback to simple centered differences
            dT = np.diff(x_data)
            dE = np.diff(y_data)
            C_mid = dE / dT
            # Interpolate back to original grid
            C = np.interp(x_data, 0.5*(x_data[:-1] + x_data[1:]), C_mid)
            return C
        
        # Fit spline to E(T)
        try:
            # Use log(T) for better behavior at low T
            log_T = np.log(x_data)
            spline = UnivariateSpline(log_T, y_data, s=smoothing, k=3)
            
            # Check monotonicity if requested
            if monotonic:
                # Evaluate derivative at many points
                log_T_dense = np.linspace(log_T[0], log_T[-1], len(log_T)*10)
                dE_dlogT = spline.derivative()(log_T_dense)
                
                # If not monotonic, increase smoothing
                if np.any(dE_dlogT < 0):
                    print(f"  Warning: E(T) not monotonic, increasing smoothing")
                    spline = UnivariateSpline(log_T, y_data, s=smoothing*5, k=3)
            
            # C(T) = dE/dT = (dE/d(logT)) / T
            dE_dlogT = spline.derivative()(log_T)
            C = dE_dlogT / x_data
            
            # Clip to ensure positivity
            C = np.maximum(C, 0.0)
            
            return C
            
        except Exception as e:
            print(f"  Warning: Spline fitting failed ({e}), using finite differences")
            dT = np.diff(x_data)
            dE = np.diff(y_data)
            C_mid = dE / dT
            C = np.interp(x_data, 0.5*(x_data[:-1] + x_data[1:]), C_mid)
            return np.maximum(C, 0.0)
    
    def check_thermodynamic_consistency(self, results, n_spins_per_unit=4, verbose=True):
        """
        Perform thermodynamic consistency checks on NLCE results.
        
        For pyrochlore tetrahedron expansion: n_spins_per_unit = 4
        
        Checks:
        1. C(T) ≥ 0 (positivity)
        2. S(∞) → n_spins * ln(2) at high T
        3. E(T) monotonically increasing
        
        Args:
            results: Dictionary with energy, specific_heat, entropy, temperatures
            n_spins_per_unit: Number of spins per expansion unit (4 for tetrahedron)
            verbose: Print diagnostic info
            
        Returns:
            Dictionary with consistency metrics
        """
        T = results['temperatures']
        E = results['energy']
        C = results['specific_heat']
        S = results['entropy']
        
        checks = {}
        
        # 1. Check C(T) ≥ 0
        C_negative = C < 0
        n_negative = np.sum(C_negative)
        checks['C_positive'] = (n_negative == 0)
        checks['C_negative_count'] = n_negative
        checks['C_min'] = np.min(C)
        
        # 2. Check E(T) monotonicity
        dE = np.diff(E)
        E_decreasing = dE < 0
        n_decreasing = np.sum(E_decreasing)
        checks['E_monotonic'] = (n_decreasing == 0)
        checks['E_nonmonotonic_count'] = n_decreasing
        
        # 3. Check high-T entropy limit
        S_infinity_expected = n_spins_per_unit * np.log(2)
        S_high_T = S[-5:]  # Last 5 temperature points
        S_avg_high_T = np.mean(S_high_T)
        S_deviation = np.abs(S_avg_high_T - S_infinity_expected) / S_infinity_expected
        
        checks['S_infinity_expected'] = S_infinity_expected
        checks['S_high_T_avg'] = S_avg_high_T
        checks['S_high_T_deviation'] = S_deviation
        checks['S_consistent'] = (S_deviation < 0.2)  # Within 20%
        
        if verbose:
            print("\n" + "="*80)
            print("THERMODYNAMIC CONSISTENCY CHECKS")
            print("="*80)
            print(f"Expansion unit: {n_spins_per_unit} spins per tetrahedron")
            print(f"")
            print(f"1. Specific Heat Positivity:")
            if checks['C_positive']:
                print(f"   ✓ C(T) ≥ 0 everywhere")
            else:
                print(f"   ✗ C(T) < 0 at {n_negative}/{len(C)} points")
                print(f"     Min C(T) = {checks['C_min']:.6e}")
            
            print(f"")
            print(f"2. Energy Monotonicity:")
            if checks['E_monotonic']:
                print(f"   ✓ E(T) monotonically increasing")
            else:
                print(f"   ✗ E(T) decreases at {n_decreasing}/{len(dE)} transitions")
            
            print(f"")
            print(f"3. High-T Entropy Limit:")
            print(f"   Expected S(∞) = {n_spins_per_unit} ln(2) = {S_infinity_expected:.6f}")
            print(f"   Observed S(high-T) = {S_avg_high_T:.6f}")
            print(f"   Relative deviation = {S_deviation*100:.2f}%")
            if checks['S_consistent']:
                print(f"   ✓ Within 20% tolerance")
            else:
                print(f"   ⚠ Deviation > 20% (check convergence at high T)")
            
            print("="*80)
        
        return checks
    
    def robust_specific_heat_pipeline(self, partial_sums_E, partial_sums_C, 
                                     resummation_method='robust', verbose=True):
        """
        Robust two-pipeline approach for specific heat:
        
        Pipeline A: Direct resummation of C(T) from cluster heat capacities
        Pipeline B: Resum E(T), then differentiate for C(T)
        
        Cross-validate and use agreement as error estimate.
        
        Args:
            partial_sums_E: List of energy partial sums [S_1^E, ..., S_n^E]
            partial_sums_C: List of heat capacity partial sums [S_1^C, ..., S_n^C]
            resummation_method: Acceleration method
            verbose: Print diagnostics
            
        Returns:
            Dictionary with:
                'C_direct': Pipeline A result
                'C_derivative': Pipeline B result  
                'C_consensus': Average of A & B
                'C_spread': |A - B| as error estimate
        """
        if verbose:
            print("\n" + "="*80)
            print("ROBUST SPECIFIC HEAT: TWO-PIPELINE CROSS-VALIDATION")
            print("="*80)
        
        # Pipeline A: Direct resummation of C(T)
        if verbose:
            print("\nPipeline A: Direct C(T) resummation")
            print("-"*80)
        resum_C = self.apply_resummation(partial_sums_C, resummation_method)
        C_direct = resum_C['value']
        C_direct_err = resum_C['error']
        
        # Pipeline B: Resum E(T), then differentiate
        if verbose:
            print("\nPipeline B: Resum E(T) → differentiate")
            print("-"*80)
        resum_E = self.apply_resummation(partial_sums_E, resummation_method)
        E_resummed = resum_E['value']
        E_err = resum_E['error']
        
        # Differentiate E(T) with shape constraint
        C_derivative = self.derivative_spline(self.temp_values, E_resummed, 
                                             smoothing=0.01, monotonic=True)
        
        # Cross-validation
        C_consensus = 0.5 * (C_direct + C_derivative)
        C_spread = np.abs(C_direct - C_derivative)
        
        # Total error: method error + pipeline spread
        C_total_err = np.sqrt(C_direct_err**2 + 0.5*C_spread**2)
        
        if verbose:
            # Report at reference temperature
            ref_idx = len(self.temp_values) // 2
            ref_T = self.temp_values[ref_idx]
            
            print(f"\nCross-validation at T = {ref_T:.4f}:")
            print(f"  Pipeline A (direct C):     {C_direct[ref_idx]:.6e} ± {C_direct_err[ref_idx]:.2e}")
            print(f"  Pipeline B (dE/dT):        {C_derivative[ref_idx]:.6e}")
            print(f"  Consensus:                 {C_consensus[ref_idx]:.6e}")
            print(f"  Spread (A-B):              {C_spread[ref_idx]:.2e}")
            print(f"  Relative disagreement:     {100*C_spread[ref_idx]/np.abs(C_consensus[ref_idx]):.2f}%")
            
            # Overall agreement metric
            mean_rel_spread = np.mean(C_spread / (np.abs(C_consensus) + 1e-15))
            print(f"\nOverall agreement: {100*(1-mean_rel_spread):.1f}%")
            if mean_rel_spread < 0.1:
                print("  ✓ Excellent agreement (< 10% spread)")
            elif mean_rel_spread < 0.3:
                print("  ✓ Good agreement (< 30% spread)")
            else:
                print("  ⚠ Large spread - check convergence")
            
            print("="*80)
        
        return {
            'C_direct': C_direct,
            'C_direct_error': C_direct_err,
            'C_derivative': C_derivative,
            'C_consensus': C_consensus,
            'C_spread': C_spread,
            'C_total_error': C_total_err,
            'E_resummed': E_resummed,
            'E_error': E_err,
            'pipeline_A_method': resum_C['method'],
            'pipeline_B_method': resum_E['method']
        }
    
    def sum_nlc(self, resummation_method='auto', order_cutoff=None, use_robust_pipeline=False, verbose=True):
        """
        Perform the NLC summation with resummation for series acceleration.
        
        The NLCE sum formula is:
            P_NLCE = Σ_c L(c) × W(c)
        where:
            L(c) = lattice constant (multiplicity) of cluster c
            W(c) = weight of cluster c = P(c) - Σ_s Y_{c,s} × W(s)
        
        For debugging convergence, we break this into partial sums by order:
            S_n = Σ_{c: order(c) ≤ n} L(c) × W(c)
        
        The increment at order n is:
            δ_n = S_n - S_{n-1} = Σ_{c: order(c) = n} L(c) × W(c)
        
        For a converging series, |δ_n| should decrease with n.
        
        Args:
            resummation_method: Method for series acceleration ('auto', 'direct', 'euler', 'wynn')
            order_cutoff: Maximum order to include in summation
            use_robust_pipeline: Use two-pipeline approach for specific heat
            verbose: If True, print detailed per-order contribution breakdown
            
        Returns:
            Dictionary with summed properties and errors, plus order-by-order data
        """
        print("\n" + "="*80)
        print(f"Performing NLC Summation (resummation: {resummation_method})")
        print("="*80)
        
        results = {
            'temperatures': self.temp_values,
            'energy': np.zeros_like(self.temp_values),
            'energy_error': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'specific_heat_error': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values),
            'entropy_error': np.zeros_like(self.temp_values),
            'free_energy': np.zeros_like(self.temp_values),
            'free_energy_error': np.zeros_like(self.temp_values)
        }
        
        # Track contributions by order for resummation
        # Store the contribution of each order (sum of all cluster weights at that order)
        order_contributions = defaultdict(lambda: {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values),
            'free_energy': np.zeros_like(self.temp_values)
        })
        
        # Track error contributions by order
        order_error_contributions = defaultdict(lambda: {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values),
            'free_energy': np.zeros_like(self.temp_values)
        })
        
        # Accumulate contributions by order
        # Each cluster's weight is multiplied by its multiplicity when adding to the sum
        for cluster_id in self.clusters:
            if not self.clusters[cluster_id].get('has_data', False):
                continue
            
            order = self.clusters[cluster_id]['order']
            multiplicity = self.clusters[cluster_id]['multiplicity']
            
            # Apply order cutoff if specified
            if order_cutoff is not None and order > order_cutoff:
                continue
            
            # Add weight * multiplicity for this cluster to its order
            # This matches the NLC_sum.py implementation
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                weight = self.weights[quantity][cluster_id]
                weight_error = self.weight_errors[quantity][cluster_id]
                
                order_contributions[order][quantity] += weight * multiplicity
                # Error propagation: multiply error by multiplicity, add in quadrature
                order_error_contributions[order][quantity] += (weight_error * multiplicity)**2
        
        # Take square root for final errors by order
        for order in order_error_contributions:
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                order_error_contributions[order][quantity] = np.sqrt(
                    order_error_contributions[order][quantity]
                )
        
        # Store order-by-order data in results for later analysis
        results['order_contributions'] = dict(order_contributions)
        results['order_errors'] = dict(order_error_contributions)
        
        # ========================================================================
        # PEDANTIC PER-ORDER CONTRIBUTION REPORT
        # ========================================================================
        if verbose:
            print("\n" + "="*120)
            print("DETAILED PER-ORDER CONTRIBUTION REPORT")
            print("="*120)
            print("Formula: Contribution(order n) = Σ_{c: order(c)=n} L(c) × W(c)")
            print("         Partial sum S_n = Σ_{k=1}^n Contribution(order k)")
            print("         Increment δ_n = S_n - S_{n-1} = Contribution(order n)")
            print("="*120)
        
        # Pick representative temperature indices for verbose output
        n_temps = len(self.temp_values)
        T_indices = {
            'low': max(0, n_temps // 10),
            'mid': n_temps // 2,
            'high': min(n_temps - 1, 9 * n_temps // 10)
        }
        
        # Store temperature indices for reference
        results['T_indices'] = {label: self.temp_values[idx] for label, idx in T_indices.items()}
        
        # Count clusters per order
        clusters_per_order = defaultdict(list)
        for cluster_id in self.clusters:
            if self.clusters[cluster_id].get('has_data', False):
                order = self.clusters[cluster_id]['order']
                if order_cutoff is None or order <= order_cutoff:
                    clusters_per_order[order].append(cluster_id)
        
        results['clusters_per_order'] = {k: len(v) for k, v in clusters_per_order.items()}
        
        max_order = max(order_contributions.keys()) if order_contributions else 0
        results['max_order'] = max_order
        
        if verbose:
            # Print per-order contributions for energy (most informative)
            print(f"\n{'─'*120}")
            print("ENERGY (E) contributions by order:")
            print(f"{'─'*120}")
            print(f"{'Order':>6s} | {'#Clusters':>9s} | {'T_low':>12s} | {'T_mid':>12s} | {'T_high':>12s} | "
                  f"{'δ_n(low)':>12s} | {'δ_n(mid)':>12s} | {'δ_n(high)':>12s} | {'Converging?':>12s}")
            print(f"{'─'*120}")
        
        prev_partial = {k: np.zeros_like(self.temp_values) for k in T_indices}
        for n in range(1, max_order + 1):
            if n not in order_contributions:
                continue
            
            n_clusters = len(clusters_per_order[n])
            contrib_n = order_contributions[n]['energy']
            
            # Compute partial sum up to order n
            partial_sum = np.zeros_like(self.temp_values)
            for order in range(1, n + 1):
                if order in order_contributions:
                    partial_sum += order_contributions[order]['energy']
            
            # Increments
            row = [f"{n:6d}", f"{n_clusters:9d}"]
            increments = {}
            for label, idx in T_indices.items():
                row.append(f"{partial_sum[idx]:+12.6f}")
            for label, idx in T_indices.items():
                delta = partial_sum[idx] - prev_partial[label][idx] if n > 1 else partial_sum[idx]
                increments[label] = delta
                row.append(f"{delta:+12.6f}")
            
            # Convergence check: |δ_n| < |δ_{n-1}|?
            if n > 1:
                # Check if magnitude is decreasing at mid temperature
                mid_idx = T_indices['mid']
                curr_mag = np.abs(increments['mid'])
                prev_contrib = order_contributions.get(n-1, {}).get('energy', np.zeros_like(self.temp_values))
                prev_mag = np.abs(prev_contrib[mid_idx]) if n > 1 else float('inf')
                converging = "✓ YES" if curr_mag < prev_mag else "✗ NO"
            else:
                converging = "—"
            row.append(f"{converging:>12s}")
            
            if verbose:
                print(" | ".join(row))
            
            # Update prev_partial
            for label, idx in T_indices.items():
                prev_partial[label][idx] = partial_sum[idx]
        
        if verbose:
            # Print per-order contributions for specific heat
            print(f"\n{'─'*120}")
            print("SPECIFIC HEAT (Cv) contributions by order:")
            print(f"{'─'*120}")
            print(f"{'Order':>6s} | {'S_n(low)':>12s} | {'S_n(mid)':>12s} | {'S_n(high)':>12s} | "
                  f"{'δ_n(low)':>12s} | {'δ_n(mid)':>12s} | {'δ_n(high)':>12s} | {'|δ_n/S_n|':>12s}")
            print(f"{'─'*120}")
        
        prev_Cv = {k: 0.0 for k in T_indices}
        for n in range(1, max_order + 1):
            if n not in order_contributions:
                continue
            
            # Compute partial sum up to order n
            partial_sum = np.zeros_like(self.temp_values)
            for order in range(1, n + 1):
                if order in order_contributions:
                    partial_sum += order_contributions[order]['specific_heat']
            
            row = [f"{n:6d}"]
            deltas = {}
            for label, idx in T_indices.items():
                row.append(f"{partial_sum[idx]:+12.6f}")
            for label, idx in T_indices.items():
                delta = partial_sum[idx] - prev_Cv[label] if n > 1 else partial_sum[idx]
                deltas[label] = delta
                row.append(f"{delta:+12.6f}")
            
            # Relative change at mid temperature
            mid_idx = T_indices['mid']
            rel_change = np.abs(deltas['mid']) / (np.abs(partial_sum[mid_idx]) + 1e-15)
            row.append(f"{rel_change:12.4f}")
            
            if verbose:
                print(" | ".join(row))
            
            for label, idx in T_indices.items():
                prev_Cv[label] = partial_sum[idx]
        
        if verbose:
            # Print entropy contributions
            print(f"\n{'─'*120}")
            print("ENTROPY (S) contributions by order:")
            print(f"{'─'*120}")
            print(f"{'Order':>6s} | {'S_n(low)':>12s} | {'S_n(mid)':>12s} | {'S_n(high)':>12s} | "
                  f"{'δ_n(low)':>12s} | {'δ_n(mid)':>12s} | {'δ_n(high)':>12s}")
            print(f"{'─'*120}")
        
        prev_S = {k: 0.0 for k in T_indices}
        for n in range(1, max_order + 1):
            if n not in order_contributions:
                continue
            
            partial_sum = np.zeros_like(self.temp_values)
            for order in range(1, n + 1):
                if order in order_contributions:
                    partial_sum += order_contributions[order]['entropy']
            
            row = [f"{n:6d}"]
            for label, idx in T_indices.items():
                row.append(f"{partial_sum[idx]:+12.6f}")
            for label, idx in T_indices.items():
                delta = partial_sum[idx] - prev_S[label] if n > 1 else partial_sum[idx]
                row.append(f"{delta:+12.6f}")
            
            if verbose:
                print(" | ".join(row))
            
            for label, idx in T_indices.items():
                prev_S[label] = partial_sum[idx]
        
        # Convergence summary
        if verbose:
            print(f"\n{'='*120}")
            print("CONVERGENCE SUMMARY")
            print(f"{'='*120}")
        
        # Check convergence of increments and store results
        convergence_info = {}
        for quantity in ['energy', 'specific_heat', 'entropy']:
            if max_order < 2:
                continue
            
            # Get last 3 increments at mid temperature
            mid_idx = T_indices['mid']
            increments = []
            prev = 0.0
            for n in range(1, max_order + 1):
                if n in order_contributions:
                    curr = 0.0
                    for o in range(1, n + 1):
                        if o in order_contributions:
                            curr += order_contributions[o][quantity][mid_idx]
                    increments.append(curr - prev)
                    prev = curr
            
            if len(increments) >= 2:
                # Check if series is converging
                last_inc = increments[-1]
                second_last_inc = increments[-2]
                ratio = np.abs(last_inc) / (np.abs(second_last_inc) + 1e-15)
                
                # Check for alternating behavior
                if len(increments) >= 3:
                    signs = np.sign(increments[-3:])
                    alternating = signs[0] * signs[1] < 0 and signs[1] * signs[2] < 0
                else:
                    alternating = False
                
                # Determine convergence status
                if ratio < 0.5:
                    status = 'converging_well'
                elif ratio < 1.0:
                    status = 'converging_slowly'
                else:
                    status = 'diverging' if not alternating else 'oscillating'
                
                convergence_info[quantity] = {
                    'last_increment': last_inc,
                    'second_last_increment': second_last_inc,
                    'ratio': ratio,
                    'alternating': alternating,
                    'status': status
                }
                
                if verbose:
                    print(f"\n  {quantity.upper()}:")
                    print(f"    Last increment (δ_{max_order}):          {last_inc:+.6f}")
                    print(f"    Second-last increment (δ_{max_order-1}): {second_last_inc:+.6f}")
                    print(f"    Ratio |δ_{max_order}|/|δ_{max_order-1}|: {ratio:.4f}")
                    print(f"    Alternating sign pattern: {'YES' if alternating else 'NO'}")
                    
                    if ratio < 0.5:
                        print(f"    Status: ✓ CONVERGING WELL (ratio < 0.5)")
                    elif ratio < 1.0:
                        print(f"    Status: ⚠ CONVERGING SLOWLY (0.5 < ratio < 1.0)")
                    else:
                        print(f"    Status: ✗ DIVERGING or OSCILLATING (ratio > 1.0)")
                        if alternating:
                            print(f"             → Euler resummation recommended for alternating series")
        
        results['convergence_info'] = convergence_info
        
        if verbose:
            print(f"\n{'='*120}")
        
        # Build partial sums by order for resummation
        # Partial sum S_n = sum of all weights with order <= n
        max_order = max(order_contributions.keys()) if order_contributions else 0
        
        # Store partial sums for all quantities
        all_partial_sums = {}
        all_partial_errors = {}
        
        for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
            partial_sums = []
            partial_errors = []
            
            # For each order, compute the sum of all contributions up to that order
            for n in range(1, max_order + 1):
                # Sum all order contributions from 1 to n
                partial_sum = np.zeros_like(self.temp_values)
                partial_error_sq = np.zeros_like(self.temp_values)
                
                for order in range(1, n + 1):
                    if order in order_contributions:
                        partial_sum += order_contributions[order][quantity]
                        partial_error_sq += order_error_contributions[order][quantity]**2
                
                partial_sums.append(partial_sum)
                partial_errors.append(np.sqrt(partial_error_sq))
            
            all_partial_sums[quantity] = partial_sums
            all_partial_errors[quantity] = partial_errors
        
        # Store partial sums for post-analysis
        results['partial_sums'] = all_partial_sums
        results['partial_errors'] = all_partial_errors
        
        # Use robust two-pipeline approach for specific heat if requested
        if use_robust_pipeline and len(all_partial_sums['energy']) > 0:
            print("\n" + "="*80)
            print("Using ROBUST TWO-PIPELINE approach for specific heat")
            print("="*80)
            
            # Apply robust specific heat pipeline
            robust_result = self.robust_specific_heat_pipeline(
                all_partial_sums['energy'],
                all_partial_sums['specific_heat'],
                resummation_method=resummation_method,
                verbose=True
            )
            
            # Use consensus specific heat
            results['specific_heat'] = robust_result['C_consensus']
            results['specific_heat_error'] = robust_result['C_total_error']
            results['specific_heat_method'] = f"robust_pipeline: {robust_result['pipeline_A_method']} + {robust_result['pipeline_B_method']}"
            results['specific_heat_direct'] = robust_result['C_direct']
            results['specific_heat_derivative'] = robust_result['C_derivative']
            results['specific_heat_spread'] = robust_result['C_spread']
            
            # Use resummed energy from Pipeline B
            results['energy'] = robust_result['E_resummed']
            results['energy_error'] = robust_result['E_error']
            results['energy_method'] = robust_result['pipeline_B_method']
            
            # Apply standard resummation to other quantities
            for quantity in ['entropy', 'free_energy']:
                partial_sums = all_partial_sums[quantity]
                partial_errors = all_partial_errors[quantity]
                
                if len(partial_sums) > 0:
                    print(f"\nApplying resummation to {quantity}...")
                    resum_result = self.apply_resummation(partial_sums, resummation_method)
                    results[quantity] = resum_result['value']
                    
                    method_error = resum_result['error']
                    order_error = partial_errors[-1]
                    results[f'{quantity}_error'] = np.sqrt(method_error**2 + order_error**2)
                    results[f'{quantity}_method'] = resum_result['method']
                    
                    if resum_result['stability']:
                        results[f'{quantity}_stability'] = resum_result['stability']
                else:
                    results[quantity] = np.zeros_like(self.temp_values)
                    results[f'{quantity}_error'] = np.zeros_like(self.temp_values)
                    results[f'{quantity}_method'] = 'none'
        
        else:
            # Standard resummation for all quantities
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                partial_sums = all_partial_sums[quantity]
                partial_errors = all_partial_errors[quantity]
                
                # Apply resummation
                if len(partial_sums) > 0:
                    print(f"\nApplying resummation to {quantity}...")
                    resum_result = self.apply_resummation(partial_sums, resummation_method)
                    results[quantity] = resum_result['value']
                    
                    # Combine resummation error with highest-order error
                    method_error = resum_result['error']
                    order_error = partial_errors[-1]
                    results[f'{quantity}_error'] = np.sqrt(method_error**2 + order_error**2)
                    
                    # Store additional diagnostics
                    results[f'{quantity}_method'] = resum_result['method']
                    if resum_result['stability']:
                        results[f'{quantity}_stability'] = resum_result['stability']
                else:
                    results[quantity] = np.zeros_like(self.temp_values)
                    results[f'{quantity}_error'] = np.zeros_like(self.temp_values)
                    results[f'{quantity}_method'] = 'none'
        
        # Print detailed order-by-order contributions with change detection
        ref_temp_idx = len(self.temp_values) // 2
        ref_temp = self.temp_values[ref_temp_idx]
        
        print(f"\n" + "="*100)
        print(f"ORDER-BY-ORDER ANALYSIS at T = {ref_temp:.4f}")
        print("="*100)
        print(f"{'Order':<8} {'Energy':<15} {'ΔE%':<12} {'Spec Heat':<15} {'ΔC%':<12} {'Entropy':<15} {'ΔS%':<12}")
        print("-" * 100)
        
        cumulative = {
            'energy': 0.0,
            'specific_heat': 0.0,
            'entropy': 0.0,
            'free_energy': 0.0
        }
        
        previous = {
            'energy': 0.0,
            'specific_heat': 0.0,
            'entropy': 0.0,
            'free_energy': 0.0
        }
        
        sorted_orders = sorted(order_contributions.keys())
        large_change_detected = False
        
        for i, order in enumerate(sorted_orders):
            contrib = order_contributions[order]
            
            # Calculate cumulative sum
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                cumulative[quantity] += contrib[quantity][ref_temp_idx]
            
            # Calculate percentage changes
            if i > 0:
                delta_E = ((cumulative['energy'] - previous['energy']) / (abs(previous['energy']) + 1e-15)) * 100
                delta_C = ((cumulative['specific_heat'] - previous['specific_heat']) / (abs(previous['specific_heat']) + 1e-15)) * 100
                delta_S = ((cumulative['entropy'] - previous['entropy']) / (abs(previous['entropy']) + 1e-15)) * 100
            else:
                delta_E = delta_C = delta_S = 0.0
            
            # Flag large changes
            warning = ""
            if abs(delta_E) > 20 or abs(delta_C) > 20 or abs(delta_S) > 20:
                warning = "  ⚠ LARGE CHANGE"
                large_change_detected = True
            
            print(f"{order:<8} {cumulative['energy']:<15.6e} {delta_E:<12.2f} "
                  f"{cumulative['specific_heat']:<15.6e} {delta_C:<12.2f} "
                  f"{cumulative['entropy']:<15.6e} {delta_S:<12.2f}{warning}")
            
            # Store previous for next iteration
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                previous[quantity] = cumulative[quantity]
        
        print("-" * 100)
        print(f"{'Direct':<8} {cumulative['energy']:<15.6e} {'Final':<12} "
              f"{cumulative['specific_heat']:<15.6e} {'Final':<12} "
              f"{cumulative['entropy']:<15.6e} {'Final':<12}")
        
        if resummation_method != 'direct':
            resum_dE = ((results['energy'][ref_temp_idx] - cumulative['energy']) / (abs(cumulative['energy']) + 1e-15)) * 100
            resum_dC = ((results['specific_heat'][ref_temp_idx] - cumulative['specific_heat']) / (abs(cumulative['specific_heat']) + 1e-15)) * 100
            resum_dS = ((results['entropy'][ref_temp_idx] - cumulative['entropy']) / (abs(cumulative['entropy']) + 1e-15)) * 100
            
            print(f"{'Resummed':<8} {results['energy'][ref_temp_idx]:<15.6e} {resum_dE:<12.2f} "
                  f"{results['specific_heat'][ref_temp_idx]:<15.6e} {resum_dC:<12.2f} "
                  f"{results['entropy'][ref_temp_idx]:<15.6e} {resum_dS:<12.2f}")
        
        print("="*100)
        
        # Detailed diagnostics if large changes detected
        if large_change_detected:
            print("\n" + "!"*100)
            print("LARGE CHANGES DETECTED BETWEEN ORDERS")
            print("!"*100)
            print("\nPossible causes:")
            print("  1. New cluster topology appears at this order (expected)")
            print("  2. Numerical instability in FTLM for larger clusters")
            print("  3. Insufficient FTLM samples or Krylov dimension")
            print("  4. Bug in cluster generation or Hamiltonian setup")
            print("  5. Series entering divergent regime (check ratio test)")
            print("\nRecommendations:")
            print("  - Check individual cluster FTLM outputs for anomalies")
            print("  - Increase --ftlm_samples and --krylov_dim")
            print("  - Verify cluster generation at this order")
            print("  - Run convergence radius analysis tool")
            print("  - Compare with lower orders to isolate problematic clusters")
            print("\nDetailed order contribution breakdown:")
            print("-" * 100)
            
            # Show increment analysis
            for i, order in enumerate(sorted_orders):
                contrib = order_contributions[order]
                print(f"\nOrder {order} contributions (increment from order {order}):")  
                print(f"  Energy:       {contrib['energy'][ref_temp_idx]:.6e}")
                print(f"  Spec Heat:    {contrib['specific_heat'][ref_temp_idx]:.6e}")
                print(f"  Entropy:      {contrib['entropy'][ref_temp_idx]:.6e}")
                print(f"  Free Energy:  {contrib['free_energy'][ref_temp_idx]:.6e}")
                
                # Check for anomalously large single-order contributions
                if i > 0:
                    prev_contrib = order_contributions[sorted_orders[i-1]]
                    ratio_E = abs(contrib['energy'][ref_temp_idx]) / (abs(prev_contrib['energy'][ref_temp_idx]) + 1e-15)
                    ratio_C = abs(contrib['specific_heat'][ref_temp_idx]) / (abs(prev_contrib['specific_heat'][ref_temp_idx]) + 1e-15)
                    
                    if ratio_E > 5 or ratio_C > 5:
                        print(f"  ⚠ INCREMENT RATIO vs Order {sorted_orders[i-1]}: Energy={ratio_E:.2f}x, SpecHeat={ratio_C:.2f}x")
                        print(f"     This order contributes significantly more than previous order!")
            
            print("\n" + "!"*100)
        
        print("="*100)
        
        return results
    
    def run(self, resummation_method='euler', order_cutoff=None, use_robust_pipeline=False, 
            n_spins_per_unit=4, verbose=True):
        """
        Run the complete NLC calculation.
        
        Args:
            resummation_method: Acceleration method
            order_cutoff: Maximum order
            use_robust_pipeline: Use two-pipeline cross-validation for C(T)
            n_spins_per_unit: Spins per expansion unit (4 for pyrochlore tetrahedron)
            verbose: Print detailed per-order contribution breakdown
        """
        self.read_clusters()
        self.read_ftlm_data()
        self.calculate_weights(verbose=verbose)
        results = self.sum_nlc(resummation_method, order_cutoff, use_robust_pipeline, verbose=verbose)
        
        # Perform thermodynamic consistency checks
        if use_robust_pipeline:
            self.check_thermodynamic_consistency(results, n_spins_per_unit, verbose=True)
        
        return results
    
    def plot_results(self, results, save_path=None):
        """Plot NLC results."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plots.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NLCE Results (FTLM)', fontsize=16, fontweight='bold')
        
        temps = results['temperatures']
        
        # Determine unit labels
        if self.SI:
            energy_unit = 'J/mol'
            cv_unit = 'J/(K·mol)'
            entropy_unit = 'J/(K·mol)'
            free_energy_unit = 'J/mol'
        else:
            energy_unit = 'natural units'
            cv_unit = 'natural units'
            entropy_unit = 'natural units'
            free_energy_unit = 'natural units'
        
        # Energy
        ax = axes[0, 0]
        ax.errorbar(temps, results['energy'], yerr=results['energy_error'],
                   fmt='o-', capsize=3, label='NLC')
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Energy per site ({energy_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Energy')
        
        # Specific Heat
        ax = axes[0, 1]
        ax.errorbar(temps, results['specific_heat'], yerr=results['specific_heat_error'],
                   fmt='o-', capsize=3, label='NLC', color='orange')
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Specific Heat per site ({cv_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Specific Heat')
        
        # Entropy
        ax = axes[1, 0]
        ax.errorbar(temps, results['entropy'], yerr=results['entropy_error'],
                   fmt='o-', capsize=3, label='NLC', color='green')
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Entropy per site ({entropy_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Entropy')
        
        # Free Energy
        ax = axes[1, 1]
        ax.errorbar(temps, results['free_energy'], yerr=results['free_energy_error'],
                   fmt='o-', capsize=3, label='NLC', color='red')
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Free Energy per site ({free_energy_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Free Energy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_specific_heat_with_experiment(self, results, exp_temp=None, exp_spec_heat=None, save_path=None):
        """Plot specific heat results with optional experimental data overlay."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return
        
        plt.figure(figsize=(10, 7))
        
        temps = results['temperatures']
        
        # Determine unit labels
        if self.SI:
            cv_unit = 'J/(K·mol)'
        else:
            cv_unit = 'natural units'
        
        # Plot NLC specific heat
        plt.plot(temps, results['specific_heat'], 'o-', color='orange', linewidth=2.5, 
                markersize=6, label='NLCE-FTLM', zorder=3)
        
        # Plot experimental data if provided
        if exp_temp is not None and exp_spec_heat is not None:
            plt.scatter(exp_temp, exp_spec_heat, color='blue', s=80, alpha=0.7, 
                       label='Experimental Data', zorder=4)
        
        plt.xlabel('Temperature (K)' if self.SI else 'Temperature', fontsize=12)
        plt.ylabel(f'Specific Heat per site ({cv_unit})', fontsize=12)
        plt.title('Specific Heat Comparison', fontsize=14, fontweight='bold')
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Specific heat plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_resummation_methods(self, partial_sums_by_quantity, save_dir=None):
        """
        Compare all resummation methods side-by-side for diagnostics.
        
        Useful for understanding which method works best for your NLCE data.
        
        Args:
            partial_sums_by_quantity: Dict of {quantity: [S_1, ..., S_N]}
            save_dir: Directory to save comparison plots/data
        """
        print("\n" + "="*80)
        print("RESUMMATION METHOD COMPARISON")
        print("="*80)
        
        methods = ['direct', 'euler', 'wynn', 'theta']
        comparison_data = {}
        
        for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
            if quantity not in partial_sums_by_quantity:
                continue
            
            partial_sums = partial_sums_by_quantity[quantity]
            if len(partial_sums) < 3:
                print(f"\n{quantity}: insufficient data (n={len(partial_sums)})")
                continue
            
            print(f"\n{quantity.upper()}:")
            print("-" * 80)
            
            comparison_data[quantity] = {}
            
            for method in methods:
                result = self.apply_resummation(partial_sums, method=method)
                comparison_data[quantity][method] = result
                
                # Print results at reference temperature
                ref_idx = len(self.temp_values) // 2
                ref_temp = self.temp_values[ref_idx]
                val = result['value'][ref_idx]
                err = result['error'][ref_idx]
                
                print(f"  {method:10s}: {val:+.8e} ± {err:.2e} (T={ref_temp:.4f})")
            
            # Analyze agreement
            values = [comparison_data[quantity][m]['value'] for m in methods]
            ref_values = [v[len(self.temp_values)//2] for v in values]
            spread = np.std(ref_values)
            mean_val = np.mean(ref_values)
            
            print(f"  {'Spread':10s}: {spread:.2e} ({100*spread/abs(mean_val):.2f}% relative)")
            
            # Convergence diagnostics
            conv = self.analyze_convergence(partial_sums)
            print(f"  Recommended: {self.select_resummation_method(partial_sums, verbose=False)}")
            print(f"  Alternating: {conv['alternating_tail']}, Converged: {conv['converged']}")
        
        # Save comparison if requested
        if save_dir:
            import json
            comparison_file = os.path.join(save_dir, "resummation_comparison.json")
            
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for quantity, methods_dict in comparison_data.items():
                json_data[quantity] = {}
                for method, result in methods_dict.items():
                    json_data[quantity][method] = {
                        'value': result['value'].tolist(),
                        'error': result['error'].tolist(),
                        'method_name': result['method']
                    }
            
            with open(comparison_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"\nComparison data saved to: {comparison_file}")
        
        print("="*80)
        return comparison_data
    
    def plot_order_by_order_thermo(self, save_dir=None):
        """Plot thermodynamic properties for each order."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available. Skipping order-by-order plots.")
            return
        
        # Determine unit labels
        if self.SI:
            energy_unit = 'J/mol'
            cv_unit = 'J/(K·mol)'
            entropy_unit = 'J/(K·mol)'
            free_energy_unit = 'J/mol'
        else:
            energy_unit = 'natural units'
            cv_unit = 'natural units'
            entropy_unit = 'natural units'
            free_energy_unit = 'natural units'
        
        # Organize clusters by order
        clusters_by_order = defaultdict(list)
        for cluster_id, data in self.clusters.items():
            if data.get('has_data', False):
                clusters_by_order[data['order']].append(cluster_id)
        
        # Get all orders
        orders = sorted(clusters_by_order.keys())
        
        if not orders:
            print("No data available for order-by-order plotting.")
            return
        
        # Compute partial sums for each order
        partial_sums = defaultdict(lambda: {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values),
            'free_energy': np.zeros_like(self.temp_values)
        })
        
        # Calculate cumulative sums up to each order
        for order in orders:
            # Sum contributions from all orders up to current
            for o in range(1, order + 1):
                if o not in clusters_by_order:
                    continue
                
                for cluster_id in clusters_by_order[o]:
                    multiplicity = self.clusters[cluster_id]['multiplicity']
                    
                    for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                        if cluster_id in self.weights[quantity]:
                            weight = self.weights[quantity][cluster_id]
                            partial_sums[order][quantity] += weight * multiplicity
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NLCE Convergence: Partial Sums by Order', fontsize=16, fontweight='bold')
        
        temps = self.temp_values
        
        # Define colors for different orders
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(orders)))
        
        # Energy
        ax = axes[0, 0]
        for idx, order in enumerate(orders):
            ax.plot(temps, partial_sums[order]['energy'], 
                   label=f'Order {order}', color=colors[idx], linewidth=2)
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Energy per site ({energy_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Energy')
        
        # Specific Heat
        ax = axes[0, 1]
        for idx, order in enumerate(orders):
            ax.plot(temps, partial_sums[order]['specific_heat'], 
                   label=f'Order {order}', color=colors[idx], linewidth=2)
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Specific Heat per site ({cv_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Specific Heat')
        
        # Entropy
        ax = axes[1, 0]
        for idx, order in enumerate(orders):
            ax.plot(temps, partial_sums[order]['entropy'], 
                   label=f'Order {order}', color=colors[idx], linewidth=2)
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Entropy per site ({entropy_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Entropy')
        
        # Free Energy
        ax = axes[1, 1]
        for idx, order in enumerate(orders):
            ax.plot(temps, partial_sums[order]['free_energy'], 
                   label=f'Order {order}', color=colors[idx], linewidth=2)
        ax.set_xlabel('Temperature (K)' if self.SI else 'Temperature')
        ax.set_ylabel(f'Free Energy per site ({free_energy_unit})')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('Free Energy')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, "nlc_order_by_order_convergence.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Order-by-order convergence plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        # Also save partial sum data to files
        if save_dir:
            for order in orders:
                order_file = os.path.join(save_dir, f"nlc_partial_sum_order_{order}.txt")
                with open(order_file, 'w') as f:
                    f.write(f"# NLCE Partial Sum up to Order {order}\n")
                    f.write(f"# Temperature  Energy  Specific_Heat  Entropy  Free_Energy\n")
                    for i, temp in enumerate(temps):
                        f.write(f"{temp:.12e}  {partial_sums[order]['energy'][i]:.12e}  "
                               f"{partial_sums[order]['specific_heat'][i]:.12e}  "
                               f"{partial_sums[order]['entropy'][i]:.12e}  "
                               f"{partial_sums[order]['free_energy'][i]:.12e}\n")
                print(f"Partial sum data for order {order} saved to: {order_file}")

    def plot_verbose_cluster_contributions(self, save_dir=None):
        """
        Create comprehensive verbose plots showing order-by-order contributions:
        1. Raw cluster properties P(c) for each cluster
        2. Cluster weights W(c) = P(c) - Σ Y_cs * W(s)
        3. Weighted contributions L(c) * W(c) per cluster
        4. Order-by-order increments δ_n = S_n - S_{n-1}
        5. Cumulative partial sums S_n
        
        This provides detailed diagnostics for understanding NLCE convergence.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
        except ImportError:
            print("Matplotlib not available. Skipping verbose plots.")
            return
        
        print("\n" + "="*100)
        print("GENERATING VERBOSE CLUSTER CONTRIBUTION PLOTS")
        print("="*100)
        
        # Organize clusters by order
        clusters_by_order = defaultdict(list)
        for cluster_id, data in self.clusters.items():
            if data.get('has_data', False):
                clusters_by_order[data['order']].append(cluster_id)
        
        orders = sorted(clusters_by_order.keys())
        if not orders:
            print("No cluster data available for plotting.")
            return
        
        max_order = max(orders)
        temps = self.temp_values
        
        # Collect data for each cluster
        cluster_data = {}
        for order in orders:
            for cluster_id in clusters_by_order[order]:
                mult = self.clusters[cluster_id]['multiplicity']
                thermo = self.clusters[cluster_id]['thermo_data']
                
                cluster_data[cluster_id] = {
                    'order': order,
                    'multiplicity': mult,
                    'P': {q: thermo[q] for q in ['energy', 'specific_heat', 'entropy', 'free_energy']},
                    'W': {q: self.weights[q].get(cluster_id, np.zeros_like(temps)) 
                          for q in ['energy', 'specific_heat', 'entropy', 'free_energy']},
                    'LW': {q: mult * self.weights[q].get(cluster_id, np.zeros_like(temps))
                           for q in ['energy', 'specific_heat', 'entropy', 'free_energy']}
                }
        
        # Compute order increments (sum of L*W for all clusters at that order)
        order_increments = defaultdict(lambda: {q: np.zeros_like(temps) 
                                                 for q in ['energy', 'specific_heat', 'entropy', 'free_energy']})
        for cluster_id, data in cluster_data.items():
            order = data['order']
            for q in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                order_increments[order][q] += data['LW'][q]
        
        # Compute cumulative partial sums
        partial_sums = defaultdict(lambda: {q: np.zeros_like(temps) 
                                             for q in ['energy', 'specific_heat', 'entropy', 'free_energy']})
        for n in orders:
            for q in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                if n == 1:
                    partial_sums[n][q] = order_increments[n][q].copy()
                else:
                    prev_order = max(o for o in orders if o < n)
                    partial_sums[n][q] = partial_sums[prev_order][q] + order_increments[n][q]
        
        # Color scheme
        n_clusters = len(cluster_data)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_clusters)))
        order_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(orders)))
        
        # ========================================================================
        # FIGURE 1: Raw cluster properties P(c) by order
        # ========================================================================
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
        fig1.suptitle('Raw Cluster Properties P(c) - Before Weight Subtraction', 
                      fontsize=16, fontweight='bold')
        
        quantities = ['energy', 'specific_heat', 'entropy', 'free_energy']
        titles = ['Energy P(c)', 'Specific Heat P(c)', 'Entropy P(c)', 'Free Energy P(c)']
        
        for ax, quantity, title in zip(axes1.flat, quantities, titles):
            color_idx = 0
            for order in orders:
                for cluster_id in clusters_by_order[order]:
                    data = cluster_data[cluster_id]
                    label = f"C{cluster_id} (n={order}, L={data['multiplicity']:.2f})"
                    ax.plot(temps, data['P'][quantity], '-', color=colors[color_idx % len(colors)],
                           linewidth=1.5, label=label, alpha=0.8)
                    color_idx += 1
            
            ax.set_xlabel('Temperature')
            ax.set_ylabel(quantity.replace('_', ' ').title())
            ax.set_xscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best', ncol=2)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_raw_properties.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 2: Cluster weights W(c) = P(c) - Σ Y_cs * W(s)
        # ========================================================================
        fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle('Cluster Weights W(c) = P(c) - Σ Y_{c,s} × W(s)', 
                      fontsize=16, fontweight='bold')
        
        titles = ['Energy W(c)', 'Specific Heat W(c)', 'Entropy W(c)', 'Free Energy W(c)']
        
        for ax, quantity, title in zip(axes2.flat, quantities, titles):
            color_idx = 0
            for order in orders:
                for cluster_id in clusters_by_order[order]:
                    data = cluster_data[cluster_id]
                    label = f"C{cluster_id} (n={order}, L={data['multiplicity']:.2f})"
                    ax.plot(temps, data['W'][quantity], '-', color=colors[color_idx % len(colors)],
                           linewidth=1.5, label=label, alpha=0.8)
                    color_idx += 1
            
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Temperature')
            ax.set_ylabel(quantity.replace('_', ' ').title())
            ax.set_xscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best', ncol=2)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_weights.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 3: Weighted contributions L(c) × W(c) per cluster
        # ========================================================================
        fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
        fig3.suptitle('Weighted Contributions L(c) × W(c) to NLCE Sum', 
                      fontsize=16, fontweight='bold')
        
        titles = ['Energy L×W', 'Specific Heat L×W', 'Entropy L×W', 'Free Energy L×W']
        
        for ax, quantity, title in zip(axes3.flat, quantities, titles):
            color_idx = 0
            for order in orders:
                for cluster_id in clusters_by_order[order]:
                    data = cluster_data[cluster_id]
                    label = f"C{cluster_id} (n={order}, L={data['multiplicity']:.2f})"
                    ax.plot(temps, data['LW'][quantity], '-', color=colors[color_idx % len(colors)],
                           linewidth=1.5, label=label, alpha=0.8)
                    color_idx += 1
            
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Temperature')
            ax.set_ylabel(quantity.replace('_', ' ').title())
            ax.set_xscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best', ncol=2)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_weighted_contributions.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 4: Order-by-order increments δ_n = Σ_{clusters at n} L(c)×W(c)
        # ========================================================================
        fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))
        fig4.suptitle('Order-by-Order Increments δₙ = S_n - S_{n-1} = Σ_{order=n} L(c)×W(c)', 
                      fontsize=16, fontweight='bold')
        
        titles = ['Energy δₙ', 'Specific Heat δₙ', 'Entropy δₙ', 'Free Energy δₙ']
        
        for ax, quantity, title in zip(axes4.flat, quantities, titles):
            for idx, order in enumerate(orders):
                label = f"Order {order}"
                ax.plot(temps, order_increments[order][quantity], '-', 
                       color=order_colors[idx], linewidth=2, label=label)
            
            ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Temperature')
            ax.set_ylabel(f'Increment δₙ ({quantity.replace("_", " ")})')
            ax.set_xscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_order_increments.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 5: Cumulative partial sums S_n
        # ========================================================================
        fig5, axes5 = plt.subplots(2, 2, figsize=(16, 12))
        fig5.suptitle('Cumulative Partial Sums Sₙ = Σ_{k=1}^{n} δₖ', 
                      fontsize=16, fontweight='bold')
        
        titles = ['Energy Sₙ', 'Specific Heat Sₙ', 'Entropy Sₙ', 'Free Energy Sₙ']
        
        for ax, quantity, title in zip(axes5.flat, quantities, titles):
            for idx, order in enumerate(orders):
                label = f"Order {order}"
                ax.plot(temps, partial_sums[order][quantity], '-', 
                       color=order_colors[idx], linewidth=2, label=label)
            
            ax.set_xlabel('Temperature')
            ax.set_ylabel(f'Partial Sum Sₙ ({quantity.replace("_", " ")})')
            ax.set_xscale('log')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_partial_sums.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 6: Comparison P(c) vs W(c) for each cluster (specific heat only)
        # ========================================================================
        n_clusters_total = sum(len(clusters_by_order[o]) for o in orders)
        if n_clusters_total <= 12:
            ncols = min(3, n_clusters_total)
            nrows = (n_clusters_total + ncols - 1) // ncols
            fig6, axes6 = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
            fig6.suptitle('Comparison: Raw P(c) vs Weight W(c) for Specific Heat', 
                          fontsize=16, fontweight='bold')
            
            if n_clusters_total == 1:
                axes6 = np.array([axes6])
            axes6_flat = axes6.flat if hasattr(axes6, 'flat') else [axes6]
            
            cluster_list = []
            for order in orders:
                cluster_list.extend(clusters_by_order[order])
            
            for ax, cluster_id in zip(axes6_flat, cluster_list):
                data = cluster_data[cluster_id]
                order = data['order']
                mult = data['multiplicity']
                
                ax.plot(temps, data['P']['specific_heat'], 'b-', linewidth=2, 
                       label=f'P(c) raw', alpha=0.8)
                ax.plot(temps, data['W']['specific_heat'], 'r-', linewidth=2, 
                       label=f'W(c) weight', alpha=0.8)
                ax.plot(temps, data['LW']['specific_heat'], 'g--', linewidth=2, 
                       label=f'L×W(c) contribution', alpha=0.8)
                
                ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
                ax.set_xlabel('Temperature')
                ax.set_ylabel('Specific Heat')
                ax.set_xscale('log')
                ax.set_title(f'Cluster {cluster_id} (order={order}, L={mult:.2f})')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            
            # Hide unused axes
            for ax in list(axes6_flat)[n_clusters_total:]:
                ax.set_visible(False)
            
            plt.tight_layout()
            if save_dir:
                path = os.path.join(save_dir, "nlc_verbose_P_vs_W_comparison.png")
                plt.savefig(path, dpi=300, bbox_inches='tight')
                print(f"Saved: {path}")
            plt.close()
        
        # ========================================================================
        # FIGURE 7: Stacked bar chart of contributions at representative temps
        # ========================================================================
        fig7, axes7 = plt.subplots(1, 3, figsize=(18, 6))
        fig7.suptitle('Stacked Contributions by Order at Representative Temperatures', 
                      fontsize=16, fontweight='bold')
        
        # Select representative temperatures
        T_indices = [len(temps)//10, len(temps)//2, 9*len(temps)//10]
        T_labels = ['Low T', 'Mid T', 'High T']
        
        for ax, T_idx, T_label in zip(axes7, T_indices, T_labels):
            T_val = temps[T_idx]
            
            # Collect contributions for specific heat
            order_vals = [order_increments[o]['specific_heat'][T_idx] for o in orders]
            partial_sum_vals = [partial_sums[o]['specific_heat'][T_idx] for o in orders]
            
            x = np.arange(len(orders))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, order_vals, width, label='Increment δₙ', 
                          color=order_colors, alpha=0.7)
            bars2 = ax.bar(x + width/2, partial_sum_vals, width, label='Partial Sum Sₙ',
                          color=order_colors, alpha=0.4, edgecolor='black', linewidth=1.5)
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Order')
            ax.set_ylabel('Specific Heat')
            ax.set_title(f'{T_label}: T = {T_val:.4f}')
            ax.set_xticks(x)
            ax.set_xticklabels([str(o) for o in orders])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars1, order_vals):
                height = bar.get_height()
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3 if height >= 0 else -12),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=7, rotation=45)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_stacked_contributions.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 8: Per-cluster contribution breakdown with formulas
        # Shows L(c) × W(c) for each cluster separately with formula annotation
        # ========================================================================
        n_clusters_total = sum(len(clusters_by_order[o]) for o in orders)
        ncols = min(4, n_clusters_total)
        nrows = (n_clusters_total + ncols - 1) // ncols
        
        fig8, axes8 = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig8.suptitle('Per-Cluster Contribution Breakdown: L(c) × W(c)\nFormula: W(c) = P(c) - Σ Y_{c,s} × W(s)', 
                      fontsize=14, fontweight='bold')
        
        if n_clusters_total == 1:
            axes8 = np.array([axes8])
        axes8_flat = axes8.flat if hasattr(axes8, 'flat') else [axes8]
        
        cluster_list = []
        for order in orders:
            cluster_list.extend(sorted(clusters_by_order[order]))
        
        for ax, cluster_id in zip(axes8_flat, cluster_list):
            data = cluster_data[cluster_id]
            order = data['order']
            mult = data['multiplicity']
            
            # Get subcluster info for this cluster
            subcluster_info = self.subcluster_info.get(cluster_id, {})
            
            # Plot the three curves
            ax.plot(temps, data['P']['specific_heat'], 'b-', linewidth=1.5, 
                   label=f'P(c)', alpha=0.7)
            ax.plot(temps, data['W']['specific_heat'], 'r-', linewidth=1.5, 
                   label=f'W(c)', alpha=0.7)
            ax.plot(temps, data['LW']['specific_heat'], 'g-', linewidth=2.5, 
                   label=f'L×W = {mult:.2f}×W', alpha=0.9)
            
            ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('T', fontsize=9)
            ax.set_ylabel('Cv contribution', fontsize=9)
            ax.set_xscale('log')
            ax.set_title(f'C{cluster_id}: order={order}, L={mult:.2f}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')
            
            # Add formula text box showing subcluster subtraction
            if subcluster_info:
                sub_str = ', '.join([f'Y_{{{s}}}={cnt}' for s, cnt in sorted(subcluster_info.items())])
                formula_text = f'W = P - [{sub_str}]×W_s'
            else:
                formula_text = 'W = P (base cluster)'
            
            # Add text box with formula
            ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, fontsize=7,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
            
            # Add value annotation at mid-temperature
            mid_idx = len(temps) // 2
            mid_T = temps[mid_idx]
            LW_mid = data['LW']['specific_heat'][mid_idx]
            ax.annotate(f'L×W={LW_mid:+.4f}', xy=(mid_T, LW_mid), 
                       xytext=(5, 10), textcoords='offset points',
                       fontsize=7, color='green',
                       arrowprops=dict(arrowstyle='->', color='green', lw=0.5))
        
        # Hide unused axes
        for ax in list(axes8_flat)[n_clusters_total:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_per_cluster_breakdown.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # FIGURE 9: Contribution magnitude comparison (sorted by |L×W| at mid-T)
        # ========================================================================
        mid_idx = len(temps) // 2
        mid_T = temps[mid_idx]
        
        # Sort clusters by absolute contribution at mid-T
        cluster_contrib = []
        for cluster_id in cluster_list:
            data = cluster_data[cluster_id]
            LW_cv = data['LW']['specific_heat'][mid_idx]
            cluster_contrib.append((cluster_id, data['order'], data['multiplicity'], LW_cv))
        
        cluster_contrib.sort(key=lambda x: abs(x[3]), reverse=True)
        
        fig9, (ax9a, ax9b) = plt.subplots(1, 2, figsize=(16, 6))
        fig9.suptitle(f'Contribution Ranking at T = {mid_T:.4f}', fontsize=14, fontweight='bold')
        
        # Left: Bar chart of contributions
        x_pos = np.arange(len(cluster_contrib))
        bar_colors = ['red' if c[3] < 0 else 'blue' for c in cluster_contrib]
        bars = ax9a.bar(x_pos, [c[3] for c in cluster_contrib], color=bar_colors, alpha=0.7)
        ax9a.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax9a.set_xlabel('Cluster (sorted by |L×W|)')
        ax9a.set_ylabel('L × W contribution to Cv')
        ax9a.set_title('Specific Heat Contributions (sorted by magnitude)')
        ax9a.set_xticks(x_pos)
        ax9a.set_xticklabels([f'C{c[0]}\n(n={c[1]})' for c in cluster_contrib], fontsize=8, rotation=45)
        ax9a.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, c in zip(bars, cluster_contrib):
            height = bar.get_height()
            ax9a.annotate(f'{height:+.4f}\nL={c[2]:.1f}',
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 5 if height >= 0 else -20),
                         textcoords="offset points",
                         ha='center', va='bottom' if height >= 0 else 'top',
                         fontsize=7)
        
        # Right: Cumulative contribution
        cumsum = np.cumsum([c[3] for c in cluster_contrib])
        ax9b.plot(x_pos, cumsum, 'bo-', linewidth=2, markersize=6)
        ax9b.fill_between(x_pos, cumsum, alpha=0.3)
        ax9b.axhline(y=cumsum[-1], color='red', linestyle='--', linewidth=1.5, 
                    label=f'Total = {cumsum[-1]:.4f}')
        ax9b.set_xlabel('Number of clusters included')
        ax9b.set_ylabel('Cumulative Cv')
        ax9b.set_title('Cumulative Sum (adding clusters by contribution magnitude)')
        ax9b.set_xticks(x_pos)
        ax9b.set_xticklabels([f'C{c[0]}' for c in cluster_contrib], fontsize=8, rotation=45)
        ax9b.grid(True, alpha=0.3)
        ax9b.legend(loc='best')
        
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, "nlc_verbose_contribution_ranking.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.close()
        
        # ========================================================================
        # Save detailed numerical data to text file
        # ========================================================================
        if save_dir:
            data_file = os.path.join(save_dir, "nlc_verbose_contribution_data.txt")
            with open(data_file, 'w') as f:
                f.write("="*120 + "\n")
                f.write("VERBOSE NLCE CONTRIBUTION DATA\n")
                f.write("="*120 + "\n\n")
                
                f.write("LEGEND:\n")
                f.write("  P(c) = Raw cluster property from FTLM\n")
                f.write("  W(c) = Weight = P(c) - Σ Y_{c,s} × W(s)\n")
                f.write("  L(c) = Multiplicity (embeddings per site)\n")
                f.write("  L×W  = Contribution to NLCE sum\n")
                f.write("  δₙ   = Order increment = Σ_{clusters at order n} L×W\n")
                f.write("  Sₙ   = Partial sum = Σ_{k=1}^{n} δₖ\n")
                f.write("\n")
                
                # Cluster-by-cluster data
                f.write("="*120 + "\n")
                f.write("PER-CLUSTER DATA\n")
                f.write("="*120 + "\n")
                
                for order in orders:
                    f.write(f"\n--- ORDER {order} ---\n")
                    for cluster_id in clusters_by_order[order]:
                        data = cluster_data[cluster_id]
                        f.write(f"\nCluster {cluster_id} (L = {data['multiplicity']:.6f}):\n")
                        f.write(f"{'T':>12s}  {'P(E)':>14s}  {'W(E)':>14s}  {'L×W(E)':>14s}  "
                               f"{'P(Cv)':>14s}  {'W(Cv)':>14s}  {'L×W(Cv)':>14s}\n")
                        f.write("-"*110 + "\n")
                        
                        # Sample every 10th temperature
                        for i in range(0, len(temps), max(1, len(temps)//10)):
                            f.write(f"{temps[i]:12.6f}  "
                                   f"{data['P']['energy'][i]:+14.6f}  "
                                   f"{data['W']['energy'][i]:+14.6f}  "
                                   f"{data['LW']['energy'][i]:+14.6f}  "
                                   f"{data['P']['specific_heat'][i]:+14.6f}  "
                                   f"{data['W']['specific_heat'][i]:+14.6f}  "
                                   f"{data['LW']['specific_heat'][i]:+14.6f}\n")
                
                # Order-by-order increments
                f.write("\n" + "="*120 + "\n")
                f.write("ORDER-BY-ORDER INCREMENTS δₙ\n")
                f.write("="*120 + "\n")
                f.write(f"{'T':>12s}  " + "  ".join([f"{'δ'+str(o)+' E':>12s}" for o in orders]) + 
                       "  " + "  ".join([f"{'δ'+str(o)+' Cv':>12s}" for o in orders]) + "\n")
                f.write("-"*120 + "\n")
                
                for i in range(0, len(temps), max(1, len(temps)//20)):
                    row = f"{temps[i]:12.6f}  "
                    row += "  ".join([f"{order_increments[o]['energy'][i]:+12.6f}" for o in orders])
                    row += "  "
                    row += "  ".join([f"{order_increments[o]['specific_heat'][i]:+12.6f}" for o in orders])
                    f.write(row + "\n")
                
                # Partial sums
                f.write("\n" + "="*120 + "\n")
                f.write("CUMULATIVE PARTIAL SUMS Sₙ\n")
                f.write("="*120 + "\n")
                f.write(f"{'T':>12s}  " + "  ".join([f"{'S'+str(o)+' E':>12s}" for o in orders]) + 
                       "  " + "  ".join([f"{'S'+str(o)+' Cv':>12s}" for o in orders]) + "\n")
                f.write("-"*120 + "\n")
                
                for i in range(0, len(temps), max(1, len(temps)//20)):
                    row = f"{temps[i]:12.6f}  "
                    row += "  ".join([f"{partial_sums[o]['energy'][i]:+12.6f}" for o in orders])
                    row += "  "
                    row += "  ".join([f"{partial_sums[o]['specific_heat'][i]:+12.6f}" for o in orders])
                    f.write(row + "\n")
                
                f.write("\n" + "="*120 + "\n")
            
            print(f"Saved: {data_file}")
        
        print("="*100)
        print("VERBOSE PLOTTING COMPLETE")
        print("="*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run NLC calculation using FTLM data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python NLC_sum_ftlm.py --cluster_dir clusters/ --ftlm_dir ftlm_results/ --output_dir nlc_output/
  python NLC_sum_ftlm.py --cluster_dir clusters/ --ftlm_dir ftlm_results/ --plot --order_cutoff 4
  python NLC_sum_ftlm.py --cluster_dir clusters/ --ftlm_dir ftlm_results/ --resummation auto
  python NLC_sum_ftlm.py --cluster_dir clusters/ --ftlm_dir ftlm_results/ --resummation wynn
        """
    )
    
    parser.add_argument('--cluster_dir', required=True, 
                       help='Directory containing cluster information files')
    parser.add_argument('--ftlm_dir', required=True, 
                       help='Directory containing FTLM output files')
    parser.add_argument('--output_dir', default='.', 
                       help='Directory to save output files')
    parser.add_argument('--exp_data', type=str, default=None,
                       help='Path to experimental specific heat data file')
    parser.add_argument('--order_cutoff', type=int, 
                       help='Maximum order to include in summation')
    parser.add_argument('--resummation', type=str, default='auto',
                       choices=['auto', 'direct', 'euler', 'wynn', 'theta', 'robust'],
                       help='Resummation method for series acceleration (default: auto)\n'
                            'auto: automatically select based on convergence\n'
                            'direct: no acceleration, use highest order\n'
                            'euler: Euler transform (best for alternating series)\n'
                            'wynn: Wynn epsilon algorithm (NLCE default)\n'
                            'theta: Brezinski theta algorithm\n'
                            'robust: run multiple methods and check agreement')
    parser.add_argument('--compare_methods', action='store_true',
                       help='Compare all resummation methods side-by-side')
    parser.add_argument('--euler_l', type=int, default=3,
                       help='Number of bare terms to keep for Euler transform (default: 3)')
    parser.add_argument('--plot', action='store_true', 
                       help='Generate plot of results')
    parser.add_argument('--SI_units', action='store_true', 
                       help='Use SI units for output')
    parser.add_argument('--temp_min', type=float, default=1e-4, 
                       help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=1.0, 
                       help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=200, 
                       help='Number of temperature points')
    parser.add_argument('--robust_pipeline', action='store_true',
                       help='Use robust two-pipeline cross-validation for C(T):\n'
                            'Pipeline A: direct C(T) resummation\n'
                            'Pipeline B: resum E(T) then differentiate\n'
                            'Reports agreement and uses spread as error estimate')
    parser.add_argument('--n_spins_per_unit', type=int, default=4,
                       help='Spins per expansion unit for thermodynamic checks\n'
                            '(default: 4 for pyrochlore tetrahedron)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Enable detailed per-order contribution output (default: True)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable detailed per-order output (overrides --verbose)')
    parser.add_argument('--verbose_plot', action='store_true',
                       help='Generate comprehensive verbose plots showing:\n'
                            '  - Raw cluster properties P(c)\n'
                            '  - Cluster weights W(c)\n'
                            '  - Weighted contributions L(c)*W(c)\n'
                            '  - Order-by-order increments δₙ\n'
                            '  - Cumulative partial sums Sₙ')
    
    args = parser.parse_args()
    
    # Handle verbose flag
    verbose = args.verbose and not args.quiet
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create NLC instance
    nlc = NLCExpansionFTLM(
        args.cluster_dir, 
        args.ftlm_dir, 
        args.temp_min, 
        args.temp_max, 
        args.temp_bins,
        args.SI_units
    )
    
    # Load clusters and data first
    nlc.read_clusters()
    nlc.read_ftlm_data()
    nlc.calculate_weights(verbose=verbose)
    
    # If comparison mode requested, do that first
    if args.compare_methods:
        print("\n" + "="*80)
        print("RUNNING RESUMMATION METHOD COMPARISON")
        print("="*80)
        
        # Build partial sums for comparison
        max_order = max(nlc.clusters[cid]['order'] for cid in nlc.clusters)
        
        from collections import defaultdict
        order_contributions = defaultdict(lambda: {
            'energy': np.zeros_like(nlc.temp_values),
            'specific_heat': np.zeros_like(nlc.temp_values),
            'entropy': np.zeros_like(nlc.temp_values),
            'free_energy': np.zeros_like(nlc.temp_values)
        })
        
        for cluster_id in nlc.clusters:
            if not nlc.clusters[cluster_id].get('has_data', False):
                continue
            
            order = nlc.clusters[cluster_id]['order']
            multiplicity = nlc.clusters[cluster_id]['multiplicity']
            
            if args.order_cutoff and order > args.order_cutoff:
                continue
            
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                weight = nlc.weights[quantity][cluster_id]
                order_contributions[order][quantity] += weight * multiplicity
        
        # Build partial sums
        partial_sums_by_quantity = {q: [] for q in ['energy', 'specific_heat', 'entropy', 'free_energy']}
        
        for n in range(1, max_order + 1):
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                partial_sum = np.zeros_like(nlc.temp_values)
                for order in range(1, n + 1):
                    if order in order_contributions:
                        partial_sum += order_contributions[order][quantity]
                partial_sums_by_quantity[quantity].append(partial_sum)
        
        # Run comparison
        nlc.compare_resummation_methods(partial_sums_by_quantity, save_dir=args.output_dir)
    
    # Run NLC calculation with selected method (and optional robust pipeline)
    results = nlc.sum_nlc(
        resummation_method=args.resummation, 
        order_cutoff=args.order_cutoff,
        use_robust_pipeline=args.robust_pipeline,
        verbose=verbose
    )
    
    # Perform thermodynamic consistency checks if robust pipeline was used
    if args.robust_pipeline:
        nlc.check_thermodynamic_consistency(
            results, 
            n_spins_per_unit=args.n_spins_per_unit, 
            verbose=True
        )
    
    # Convert to SI units if requested (J/K per mol)
    if args.SI_units:
        # Conversion factor: N_A * k_B = 8.314 J/(K·mol)
        # This converts from natural units (k_B = 1) to J/K per mol
        SI_conversion = 6.02214076e23 * 1.380649e-23  # N_A * k_B in J/K
        
        print(f"\nConverting to SI units (J/K per mol)...")
        print(f"Conversion factor: N_A × k_B = {SI_conversion:.6f} J/(K·mol)")
        
        # Apply conversion to thermodynamic quantities
        for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
            results[quantity] *= SI_conversion
            results[quantity + '_error'] *= SI_conversion
    
    # Save results
    output_files = {
        'energy': os.path.join(args.output_dir, "nlc_energy.txt"),
        'specific_heat': os.path.join(args.output_dir, "nlc_specific_heat.txt"),
        'entropy': os.path.join(args.output_dir, "nlc_entropy.txt"),
        'free_energy': os.path.join(args.output_dir, "nlc_free_energy.txt")
    }
    
    # Prepare unit labels
    if args.SI_units:
        unit_labels = {
            'energy': '(J/mol)',
            'specific_heat': '(J/(K·mol))',
            'entropy': '(J/(K·mol))',
            'free_energy': '(J/mol)'
        }
    else:
        unit_labels = {
            'energy': '(natural units)',
            'specific_heat': '(natural units)',
            'entropy': '(natural units)',
            'free_energy': '(natural units)'
        }
    
    for quantity, filepath in output_files.items():
        with open(filepath, 'w') as f:
            f.write(f"# Temperature  {quantity.replace('_', ' ').title()} {unit_labels[quantity]}  Error\n")
            for i, temp in enumerate(results['temperatures']):
                f.write(f"{temp:.12e}  {results[quantity][i]:.12e}  "
                       f"{results[quantity + '_error'][i]:.12e}\n")
        print(f"Saved {quantity} to: {filepath}")
    
    # Plot if requested
    if args.plot:
        plot_path = os.path.join(args.output_dir, "nlc_results_ftlm.png")
        nlc.plot_results(results, save_path=plot_path)
        
        # Plot order-by-order convergence
        nlc.plot_order_by_order_thermo(save_dir=args.output_dir)
        
        # Generate verbose cluster contribution plots if requested
        if args.verbose_plot:
            nlc.plot_verbose_cluster_contributions(save_dir=args.output_dir)
        
        # Plot specific heat with experimental data overlay if available
        exp_temp = None
        exp_spec_heat = None
        
        if args.exp_data:
            try:
                exp_data = np.loadtxt(args.exp_data)
                exp_temp = exp_data[:, 0]
                exp_spec_heat = exp_data[:, 1]
                print(f"\nLoaded experimental data from {args.exp_data}")
            except Exception as e:
                print(f"Warning: Could not load experimental data: {e}")
        
        # Plot specific heat comparison
        spec_heat_plot_path = os.path.join(args.output_dir, "nlc_specific_heat_comparison.png")
        nlc.plot_specific_heat_with_experiment(results, exp_temp, exp_spec_heat, 
                                               save_path=spec_heat_plot_path)
    
    print(f"\nNLC calculation completed! Results saved to {args.output_dir}")
