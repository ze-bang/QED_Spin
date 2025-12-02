#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation for FTLM results.

This script performs NLCE summation using thermodynamic data obtained from
Finite Temperature Lanczos Method (FTLM) calculations on each cluster.

Key differences from standard NLC_sum.py:
- Reads FTLM output files (ftlm_thermo.txt) instead of eigenvalue files
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
    
    def read_ftlm_data(self):
        """Read FTLM thermodynamic data for each cluster."""
        print("\nReading FTLM data...")
        
        for cluster_id in self.clusters:
            order = self.clusters[cluster_id]['order']
            
            # Construct path to FTLM output
            ftlm_file = os.path.join(
                self.ftlm_dir, 
                f'cluster_{cluster_id}_order_{order}',
                'output', 'thermo', 'ftlm_thermo.txt'
            )
            
            if not os.path.exists(ftlm_file):
                print(f"Warning: FTLM data not found for cluster {cluster_id}: {ftlm_file}")
                self.clusters[cluster_id]['has_data'] = False
                continue
            
            # Read FTLM output file
            # Format: Temperature  Energy  E_error  Specific_Heat  C_error  Entropy  S_error  Free_Energy  F_error
            try:
                data = np.loadtxt(ftlm_file)
                
                if data.shape[0] != len(self.temp_values):
                    print(f"Warning: Temperature grid mismatch for cluster {cluster_id}")
                    print(f"Expected {len(self.temp_values)} points, got {data.shape[0]}")
                    # Interpolate to match expected grid
                    from scipy.interpolate import interp1d
                    
                    temp_read = data[:, 0]
                    energy = interp1d(temp_read, data[:, 1], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    energy_err = interp1d(temp_read, data[:, 2], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    spec_heat = interp1d(temp_read, data[:, 3], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    spec_heat_err = interp1d(temp_read, data[:, 4], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    entropy = interp1d(temp_read, data[:, 5], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    entropy_err = interp1d(temp_read, data[:, 6], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    free_energy = interp1d(temp_read, data[:, 7], kind='cubic', fill_value='extrapolate')(self.temp_values)
                    free_energy_err = interp1d(temp_read, data[:, 8], kind='cubic', fill_value='extrapolate')(self.temp_values)
                else:
                    energy = data[:, 1]
                    energy_err = data[:, 2]
                    spec_heat = data[:, 3]
                    spec_heat_err = data[:, 4]
                    entropy = data[:, 5]
                    entropy_err = data[:, 6]
                    free_energy = data[:, 7]
                    free_energy_err = data[:, 8]
                
                self.clusters[cluster_id]['thermo_data'] = {
                    'energy': energy,
                    'energy_error': energy_err,
                    'specific_heat': spec_heat,
                    'specific_heat_error': spec_heat_err,
                    'entropy': entropy,
                    'entropy_error': entropy_err,
                    'free_energy': free_energy,
                    'free_energy_error': free_energy_err
                }
                self.clusters[cluster_id]['has_data'] = True
                
                print(f"  Cluster {cluster_id} (order {order}): loaded {len(self.temp_values)} temperature points")
                
            except Exception as e:
                print(f"Error reading FTLM data for cluster {cluster_id}: {e}")
                self.clusters[cluster_id]['has_data'] = False
    
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
    
    def calculate_weights(self):
        """Calculate weights for all clusters using the NLC principle."""
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
        
        print("\nCalculating NLC weights...")
        
        for cluster_id, order in sorted_clusters:
            if not self.clusters[cluster_id].get('has_data', False):
                print(f"  Cluster {cluster_id} (order {order}): skipping (no data)")
                continue
            
            multiplicity = self.clusters[cluster_id]['multiplicity']
            subclusters = self.get_subclusters(cluster_id)
            
            # Get thermodynamic data for this cluster
            thermo = self.clusters[cluster_id]['thermo_data']
            
            # For each quantity, weight = cluster_value - sum of (subcluster_multiplicity * subcluster_weight)
            # NOTE: We do NOT multiply by cluster multiplicity here - that happens in sum_nlc()
            # Error propagation: σ²_weight = σ²_cluster + Σ(sub_mult² * σ²_subcluster_weight)
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                cluster_value = thermo[quantity]
                cluster_error = thermo[f'{quantity}_error']
                
                weight_value = cluster_value.copy()
                weight_error_sq = cluster_error**2
                
                # Subtract weighted contributions from subclusters
                for sub_id, sub_mult in subclusters.items():
                    if sub_id in self.weights[quantity]:
                        weight_value -= sub_mult * self.weights[quantity][sub_id]
                        # Add subcluster weight error in quadrature
                        weight_error_sq += (sub_mult * self.weight_errors[quantity][sub_id])**2
                
                self.weights[quantity][cluster_id] = weight_value
                self.weight_errors[quantity][cluster_id] = np.sqrt(weight_error_sq)
            
            print(f"  Cluster {cluster_id} (order {order}): multiplicity={multiplicity}, "
                  f"{len(subclusters)} subclusters")
    
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
        Apply Wynn's epsilon algorithm for series acceleration.
        
        Implementation follows standard NLCE practice (arXiv:1207.3366):
        Build ε table recursively, use EVEN entries only (odds diverge).
        
        Args:
            sequence: Array of partial sums S_n at each order
            return_all_evens: If True, return all even ε entries for stability analysis
            
        Returns:
            If return_all_evens: list of even ε entries (ε_0, ε_2, ε_4, ...)
            Otherwise: tuple of (best_estimate, error_estimate)
        """
        n = len(sequence)
        if n < 3:
            result = sequence[-1] if n > 0 else np.zeros_like(self.temp_values)
            if return_all_evens:
                return [result]
            return result, np.zeros_like(self.temp_values)
        
        # Guard threshold for near-singular denominators
        eps_den = 1e-14
        
        # Initialize
        eps_prev = np.zeros((n, len(self.temp_values)))  # ε_{-1}^{(n)}
        eps_curr = np.array(sequence)                     # ε_{0}^{(n)}
        
        evens = [eps_curr[0].copy()]  # Collect ε_{2m}^{(lowest n)}
        
        # Iterative construction of ε table
        iteration = 0
        while eps_curr.shape[0] > 1:
            iteration += 1
            n_curr = eps_curr.shape[0]
            eps_next = np.zeros((n_curr - 1, len(self.temp_values)))
            
            for i in range(n_curr - 1):
                denom = eps_curr[i+1] - eps_curr[i]
                
                # Handle near-singular denominators
                mask = np.abs(denom) < eps_den
                
                # Standard ε recursion: ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1/(ε_k^{(n+1)} - ε_k^{(n)})
                eps_next[i] = np.where(
                    mask,
                    eps_curr[i+1],  # Fallback: use neighbor
                    eps_prev[i+1] + 1.0 / denom
                )
            
            # Check for complete breakdown (all entries invalid)
            if np.all(np.isnan(eps_next)) or np.all(np.isinf(eps_next)):
                break
            
            # Update for next iteration
            eps_prev = eps_curr
            eps_curr = eps_next
            
            # Store even entries (iteration 1 → ε_2, iteration 3 → ε_4, etc.)
            if iteration % 2 == 1:  # Odd iterations produce even ε
                evens.append(eps_curr[0].copy())
        
        if return_all_evens:
            return evens
        
        # Use last two even entries for central value ± error
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
        
        # Initialize θ table (similar structure to ε)
        theta_prev = np.zeros((n, len(self.temp_values)))
        theta_curr = np.array(sequence)
        
        evens = [theta_curr[0].copy()]
        
        iteration = 0
        while theta_curr.shape[0] > 1:
            iteration += 1
            n_curr = theta_curr.shape[0]
            theta_next = np.zeros((n_curr - 1, len(self.temp_values)))
            
            for i in range(n_curr - 1):
                # θ recursion: θ_{k+1}^{(n)} = θ_{k-1}^{(n+1)} + (n+1)/(θ_k^{(n+1)} - θ_k^{(n)})
                denom = theta_curr[i+1] - theta_curr[i]
                mask = np.abs(denom) < eps_den
                
                # The (n+1) factor differentiates θ from ε
                theta_next[i] = np.where(
                    mask,
                    theta_curr[i+1],
                    theta_prev[i+1] + (i + 1) / denom
                )
            
            if np.all(np.isnan(theta_next)) or np.all(np.isinf(theta_next)):
                break
            
            theta_prev = theta_curr
            theta_curr = theta_next
            
            if iteration % 2 == 1:
                evens.append(theta_curr[0].copy())
        
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
            # Check if signs consistently alternate across all temperatures
            signs = np.sign(np.mean(recent_increments, axis=1))
            alternating_tail = np.all(signs[1:] * signs[:-1] < 0)
        elif n >= 3:
            signs = np.sign(np.mean(increments, axis=1))
            alternating_tail = np.all(signs[1:] * signs[:-1] < 0)
        else:
            alternating_tail = False
        
        # 2) Convergence test (magnitudes decreasing)
        if n > 3:
            recent_magnitudes = np.abs(np.mean(increments[-3:], axis=1))
            converged = (recent_magnitudes[-1] <= recent_magnitudes[-2] and 
                        recent_magnitudes[-2] <= recent_magnitudes[-3])
        else:
            converged = False
        
        # 3) Oscillatory (weaker than alternating)
        if n > 4:
            signs = np.sign(np.mean(increments[-4:], axis=1))
            oscillatory = np.sum(signs[1:] * signs[:-1] < 0) >= 2
        else:
            oscillatory = False
        
        # 4) Ratio test for convergence rate
        if n > 2:
            # |a_{n+1}| / |a_n|
            ratios = np.abs(increments[1:]) / (np.abs(increments[:-1]) + 1e-15)
            # Average ratio over last few terms and over temperatures
            avg_ratio = np.mean(ratios[-min(3, len(ratios)):])
        else:
            avg_ratio = None
        
        # 5) Monotonicity (all increments same sign)
        if n > 2:
            monotonic = np.all(np.sign(np.mean(increments, axis=1)) == np.sign(np.mean(increments[0], axis=0)))
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
        Automatically select best resummation method using convergence analysis.
        
        Decision tree follows NLCE best practices:
        - Alternating tail → Euler (designed for this)
        - Non-alternating, enough terms → Wynn ε (NLCE default)
        - Unstable/noisy → try Brezinski θ
        - Converged → direct
        
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
            print(f"    Convergence analysis: n={n}, converged={conv['converged']}, "
                  f"alternating={conv['alternating_tail']}, ratio={ratio_str}")
        
        # 1) Already converged and stable → direct
        if conv['converged'] and not conv['oscillatory'] and conv['ratio_test'] and conv['ratio_test'] < 0.3:
            return 'direct'
        
        # 2) Alternating tail detected → Euler (Tang–Khatami–Rigol recommendation)
        if conv['alternating_tail']:
            return 'euler'
        
        # 3) Too few terms for non-linear methods
        if n <= 3:
            return 'direct'
        
        # 4) Enough terms and not alternating → Wynn ε (NLCE workhorse)
        if n >= 5 and not conv['alternating_tail']:
            return 'wynn'
        
        # 5) Oscillatory but not cleanly alternating → try Brezinski θ
        if conv['oscillatory'] and n >= 5:
            return 'theta'
        
        # 6) Moderate number of terms (4-5), try Euler
        if 4 <= n < 5:
            return 'euler'
        
        # 7) Default: Wynn if enough terms, else direct
        return 'wynn' if n >= 5 else 'direct'
    
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
        if method == 'robust' and n >= 5:
            conv = self.analyze_convergence(partial_sums)
            
            results = {}
            
            # For alternating series: SKIP Wynn (it blows up), use Euler + theta
            if conv['alternating_tail']:
                euler_val, euler_err = self.euler_resummation(partial_sums, l=l_euler)
                results['euler'] = (euler_val, euler_err)
                
                # Brezinski can handle alternating better than Wynn
                theta_val, theta_err = self.brezinski_theta(partial_sums)
                results['theta'] = (theta_val, theta_err)
            else:
                # For non-alternating: use Wynn + theta
                wynn_val, wynn_err = self.wynn_epsilon(partial_sums)
                results['wynn'] = (wynn_val, wynn_err)
                
                theta_val, theta_err = self.brezinski_theta(partial_sums)
                results['theta'] = (theta_val, theta_err)
            
            # Include direct sum
            direct = partial_sums[-1]
            results['direct'] = (direct, np.abs(partial_sums[-1] - partial_sums[-2]) if n > 1 else np.zeros_like(direct))
            
            # Check agreement among methods
            all_values = [v[0] for v in results.values()]
            all_errors = [v[1] for v in results.values()]
            
            # Central value: average of the two main accelerators (excluding direct)
            accelerator_values = [v[0] for k, v in results.items() if k != 'direct']
            best_val = np.mean(accelerator_values, axis=0)
            
            # Error: spread among methods + individual errors
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
        elif method == 'theta':
            val, err = self.brezinski_theta(partial_sums)
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
    
    def sum_nlc(self, resummation_method='auto', order_cutoff=None, use_robust_pipeline=False):
        """
        Perform the NLC summation with resummation for series acceleration.
        
        Args:
            resummation_method: Method for series acceleration ('auto', 'direct', 'euler', 'wynn')
            order_cutoff: Maximum order to include in summation
            
        Returns:
            Dictionary with summed properties and errors
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
        
        # Print order-by-order contributions at a reference temperature
        ref_temp_idx = len(self.temp_values) // 2
        ref_temp = self.temp_values[ref_temp_idx]
        
        print(f"\nOrder-by-order contributions at T = {ref_temp:.4f}:")
        print("-" * 80)
        print(f"{'Order':<8} {'Energy':<15} {'Spec Heat':<15} {'Entropy':<15} {'Free Energy':<15}")
        print("-" * 80)
        
        cumulative = {
            'energy': 0.0,
            'specific_heat': 0.0,
            'entropy': 0.0,
            'free_energy': 0.0
        }
        
        for order in sorted(order_contributions.keys()):
            contrib = order_contributions[order]
            print(f"{order:<8} {contrib['energy'][ref_temp_idx]:<15.6e} "
                  f"{contrib['specific_heat'][ref_temp_idx]:<15.6e} "
                  f"{contrib['entropy'][ref_temp_idx]:<15.6e} "
                  f"{contrib['free_energy'][ref_temp_idx]:<15.6e}")
            
            for quantity in ['energy', 'specific_heat', 'entropy', 'free_energy']:
                cumulative[quantity] += contrib[quantity][ref_temp_idx]
        
        print("-" * 80)
        print(f"{'Direct':<8} {cumulative['energy']:<15.6e} "
              f"{cumulative['specific_heat']:<15.6e} "
              f"{cumulative['entropy']:<15.6e} "
              f"{cumulative['free_energy']:<15.6e}")
        
        if resummation_method != 'direct':
            print(f"{'Resummed':<8} {results['energy'][ref_temp_idx]:<15.6e} "
                  f"{results['specific_heat'][ref_temp_idx]:<15.6e} "
                  f"{results['entropy'][ref_temp_idx]:<15.6e} "
                  f"{results['free_energy'][ref_temp_idx]:<15.6e}")
        
        print("="*80)
        
        return results
    
    def run(self, resummation_method='euler', order_cutoff=None, use_robust_pipeline=False, 
            n_spins_per_unit=4):
        """
        Run the complete NLC calculation.
        
        Args:
            resummation_method: Acceleration method
            order_cutoff: Maximum order
            use_robust_pipeline: Use two-pipeline cross-validation for C(T)
            n_spins_per_unit: Spins per expansion unit (4 for pyrochlore tetrahedron)
        """
        self.read_clusters()
        self.read_ftlm_data()
        self.calculate_weights()
        results = self.sum_nlc(resummation_method, order_cutoff, use_robust_pipeline)
        
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
    
    args = parser.parse_args()
    
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
    nlc.calculate_weights()
    
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
        use_robust_pipeline=args.robust_pipeline
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
