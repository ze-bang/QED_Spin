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
                    if line.startswith("# Multiplicity:"):
                        multiplicity = float(line.split(":")[1].strip())
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
    
    def euler_resummation(self, partial_sums):
        """
        Apply Euler transformation for series acceleration.
        
        Args:
            partial_sums: Array of partial sums S_n = sum_{k=0}^n a_k
            
        Returns:
            Accelerated series approximation
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else np.zeros_like(self.temp_values)
            
        # Create difference table
        diff_table = [partial_sums[0]]
        for i in range(1, n):
            diff_table.append(partial_sums[i] - partial_sums[i-1])
            
        # Apply Euler transformation: E_n = (1/2^n) * sum_{k=0}^n C(n,k) * diff_k
        euler_sum = np.zeros_like(partial_sums[-1])
        
        for k in range(n):
            # Binomial coefficient C(n-1, k)
            binomial_coeff = 1
            for j in range(k):
                binomial_coeff = binomial_coeff * (n - 1 - j) // (j + 1)
            
            euler_sum += binomial_coeff * diff_table[k]
            
        return euler_sum / (2**(n-1))
    
    def wynn_epsilon(self, sequence):
        """
        Apply Wynn's epsilon algorithm for series acceleration.
        
        Args:
            sequence: Array of partial sums or sequence values
            
        Returns:
            Accelerated approximation using epsilon algorithm
        """
        n = len(sequence)
        if n < 3:
            return sequence[-1] if n > 0 else np.zeros_like(self.temp_values)
            
        # Initialize epsilon table
        eps = np.zeros((n + 1, n + 1, len(self.temp_values)))
            
        # Set initial values
        eps[0, :] = 0.0
        for i in range(n):
            eps[1, i] = sequence[i]
            
        # Fill epsilon table
        for k in range(2, n + 1):
            for i in range(n - k + 1):
                denominator = eps[k-1, i+1] - eps[k-1, i]
                # Avoid division by zero
                mask = np.abs(denominator) < 1e-15
                eps[k, i] = np.where(
                    mask,
                    eps[k-1, i+1],
                    eps[k-2, i+1] + 1.0 / denominator
                )
                    
        # Return the most accelerated value (highest even k)
        max_even_k = 2 * ((n) // 2)
        if max_even_k > 0:
            return eps[max_even_k, 0]
        else:
            return sequence[-1]
    
    def analyze_convergence(self, partial_sums):
        """
        Analyze the convergence behavior of the NLC series.
        
        Args:
            partial_sums: List of partial sums by order
            
        Returns:
            Dictionary with convergence diagnostics
        """
        n = len(partial_sums)
        if n < 2:
            return {'converged': False, 'oscillatory': False, 'ratio_test': None}
        
        # Convert to array for easier manipulation
        partial_sums = np.array(partial_sums)
        
        # Calculate differences between successive orders
        diffs = np.diff(partial_sums, axis=0)
        
        # Check for convergence (differences getting smaller)
        if n > 3:
            recent_diffs = np.abs(diffs[-3:])
            # Check if magnitudes are decreasing
            converged = np.all(
                np.mean(recent_diffs[-1]) <= np.mean(recent_diffs[-2])
            ) and np.all(
                np.mean(recent_diffs[-2]) <= np.mean(recent_diffs[-3])
            )
        else:
            converged = False
            
        # Check for oscillatory behavior
        if n > 4:
            # Check if signs alternate in the differences
            signs = np.sign(np.mean(diffs[-4:], axis=1))
            oscillatory = np.all(signs[1:] != signs[:-1])
        else:
            oscillatory = False
            
        # Ratio test for convergence
        if n > 2:
            ratios = np.abs(diffs[1:] / (diffs[:-1] + 1e-15))
            avg_ratio = np.mean(ratios[-min(3, len(ratios)):])
        else:
            avg_ratio = None
            
        return {
            'converged': converged,
            'oscillatory': oscillatory, 
            'ratio_test': avg_ratio
        }
    
    def select_resummation_method(self, partial_sums):
        """
        Automatically select the best resummation method based on convergence analysis.
        
        Args:
            partial_sums: List of partial sums for analysis
            
        Returns:
            String indicating the selected method
        """
        convergence = self.analyze_convergence(partial_sums)
        n = len(partial_sums)
        
        # If already converged and not oscillatory, use direct sum
        if convergence['converged'] and not convergence['oscillatory']:
            return 'direct'
            
        # For oscillatory series, Euler works well
        if convergence['oscillatory']:
            return 'euler'
            
        # For small number of terms, use direct or Euler
        if n <= 3:
            return 'direct'
        
        # For slowly converging series, try Wynn's epsilon
        if convergence['ratio_test'] is not None and convergence['ratio_test'] > 0.5:
            if n >= 5:  # Wynn needs at least 5 terms for stability
                return 'wynn'
            else:
                return 'euler'
            
        # Default to Wynn's epsilon for general case with enough terms
        if n >= 5:
            return 'wynn'
        else:
            return 'euler'
    
    def apply_resummation(self, partial_sums, method='auto'):
        """
        Apply the specified resummation method.
        
        Args:
            partial_sums: List of partial sums (one per order)
            method: Resummation method ('auto', 'direct', 'euler', 'wynn')
            
        Returns:
            Accelerated series approximation
        """
        if len(partial_sums) == 0:
            return np.zeros_like(self.temp_values)
            
        if method == 'auto':
            selected_method = self.select_resummation_method(partial_sums)
            print(f"  Auto-selected resummation method: {selected_method}")
            method = selected_method
            
        if method == 'direct':
            return partial_sums[-1]
        elif method == 'euler':
            return self.euler_resummation(partial_sums)
        elif method == 'wynn':
            return self.wynn_epsilon(partial_sums)
        else:
            print(f"  Unknown method '{method}', using direct summation")
            return partial_sums[-1]
    
    def sum_nlc(self, resummation_method='auto', order_cutoff=None):
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
            
            # Apply resummation
            if len(partial_sums) > 0:
                print(f"\nApplying resummation to {quantity}...")
                results[quantity] = self.apply_resummation(partial_sums, resummation_method)
                # Use error from highest order (most complete calculation)
                results[f'{quantity}_error'] = partial_errors[-1]
            else:
                results[quantity] = np.zeros_like(self.temp_values)
                results[f'{quantity}_error'] = np.zeros_like(self.temp_values)
        
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
    
    def run(self, resummation_method='euler', order_cutoff=None):
        """Run the complete NLC calculation."""
        self.read_clusters()
        self.read_ftlm_data()
        self.calculate_weights()
        results = self.sum_nlc(resummation_method, order_cutoff)
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
                       choices=['auto', 'direct', 'euler', 'wynn'],
                       help='Resummation method for series acceleration (default: auto)')
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
    
    # Run NLC calculation
    results = nlc.run(resummation_method=args.resummation, order_cutoff=args.order_cutoff)
    
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
