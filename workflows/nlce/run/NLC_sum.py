import os
import numpy as np
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
import argparse

#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation utility.
Calculates thermodynamic properties of a lattice using cluster expansion.
"""

import matplotlib.pyplot as plt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. HDF5 file reading will not be available.")

class NLCExpansion:
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min, temp_max, num_temps, measure_spin, SI_units=False):
        """
        Initialize the NLC expansion calculator.
        
        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_values: Array of inverse temperature values to evaluate
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir
        
        self.SI = SI_units  # Flag for SI units
        self.measure_spin = measure_spin  # Flag for measuring spin expectation values
        self.temp_values = np.logspace(np.log(temp_min)/np.log(10), np.log(temp_max)/np.log(10), num_temps)  # Default temperature range

        

        self.clusters = {}  # Will store {cluster_id: {order, multiplicity, eigenvalues, etc.}}
        self.weights = {}   # Will store calculated weights for each cluster and property
        
    def read_clusters(self):
        """Read all cluster information from files in the cluster directory."""
        pattern = os.path.join(self.cluster_dir, "cluster_*_order_*.dat")
        cluster_files = glob.glob(pattern)
        
        for file_path in cluster_files:
            # Extract cluster ID and order from filename
            match = re.search(r'cluster_(\d+)_order_(\d+)', file_path)
            if not match:
                continue
                
            cluster_id, order = int(match.group(1)), int(match.group(2))
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            multiplicity = None
            num_vertices = None
            for line in lines:
                # Handle both "# Multiplicity:" and "# Multiplicity (lattice constant):"
                if line.startswith("# Multiplicity") and ":" in line:
                    multiplicity = float(line.split(":")[-1].strip())
                elif line.startswith("# Number of vertices:"):
                    num_vertices = int(line.split(":")[1].strip())
                
                # Break if we found both values
                if multiplicity is not None and num_vertices is not None:
                    break
            
            if multiplicity is None:
                print(f"Warning: Multiplicity not found for cluster {cluster_id}")
                continue
                
            if num_vertices is None:
                print(f"Warning: Number of vertices not found for cluster {cluster_id}")
                continue
                
            self.clusters[cluster_id] = {
                'order': order,
                'multiplicity': multiplicity,
                'num_vertices': num_vertices,
                'file_path': file_path,
                'eigenvalues': None,  # Will be loaded later
                'sp': None,  # Will be loaded later
                'sm': None,  # Will be loaded later
                'sp': None   # Will be loaded later    
            }
    def read_eigenvalues(self):
        """Read eigenvalues for each cluster from ED output files (HDF5 format)."""
        for cluster_id in self.clusters:
            cluster_output_dir = os.path.join(
                self.eigenvalue_dir, 
                f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}/output"
            )
            
            # Try HDF5 file first (new format)
            h5_file = os.path.join(cluster_output_dir, "ed_results.h5")
            if HAS_H5PY and os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        if '/eigendata/eigenvalues' in f:
                            eigenvalues = f['/eigendata/eigenvalues'][:]
                            self.clusters[cluster_id]['eigenvalues'] = np.array(eigenvalues)
                            continue
                except Exception as e:
                    print(f"Warning: Error reading HDF5 file for cluster {cluster_id}: {e}")
            
            # Fall back to legacy text file format
            eigenvalue_file = os.path.join(cluster_output_dir, "eigenvalues.txt")
            if os.path.exists(eigenvalue_file):
                with open(eigenvalue_file, 'r') as f:
                    eigenvalues = [float(line.strip()) for line in f if line.strip()]
                self.clusters[cluster_id]['eigenvalues'] = np.array(eigenvalues)
                continue
            
            print(f"Warning: Eigenvalue data not found for cluster {cluster_id}")
    
    def get_subclusters(self, cluster_id):
        """
        Get all subclusters of a given cluster.
        For this demo, we'll just use a simple heuristic based on order.
        
        In a full implementation, this would use the actual topology data.
        """
        order = self.clusters[cluster_id]['order']
        return [cid for cid, data in self.clusters.items() 
                if data['order'] < order]
    
    def calculate_thermodynamic_quantities(self, eigenvalues):
        """
        Calculate thermodynamic quantities from eigenvalues.
        Uses a numerically stable approach to handle large beta values (low temperatures).
        
        Returns:
            Dictionary with 'energy', 'specific_heat', and 'entropy' as keys
        """
        results = {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values)
        }
        
        for i, temp in enumerate(self.temp_values):
            # For extremely large beta (low T), use ground state approximation
            if temp < 1e-5:
                ground_state_energy = np.min(eigenvalues)
                results['energy'][i] = ground_state_energy
                results['specific_heat'][i] = 0.0  # Specific heat approaches 0 as T->0
                results['entropy'][i] = 0.0  # Entropy approaches 0 as T->0 (third law)
                continue
                
            # Find ground state energy (minimum eigenvalue)
            ground_state_energy = np.min(eigenvalues)
            
            # Shift eigenvalues by ground state energy for numerical stability
            shifted_eigenvalues = eigenvalues - ground_state_energy
            
            # Calculate exponential terms with shifted eigenvalues
            # exp(-β(Ei-E0)) instead of exp(-βEi) to prevent underflow
            exp_terms = np.exp(-shifted_eigenvalues / temp)
            Z_shifted = np.sum(exp_terms)
            
            # Calculate energy using original eigenvalues but stable exponentials
            energy = np.sum(eigenvalues * exp_terms) / Z_shifted
            
            # Calculate energy squared
            energy_squared = np.sum(eigenvalues**2 * exp_terms) / Z_shifted
            
            # Specific heat = β² * (⟨E²⟩ - ⟨E⟩²)
            specific_heat = (energy_squared - energy**2) / (temp * temp)
            # Calculate entropy, accounting for shifted partition function
            # S = kB * [ln(Z) + βE]
            # where ln(Z) = ln(Z_shifted) + β*ground_state_energy
            entropy = (np.log(Z_shifted) + (energy - ground_state_energy) / (temp))
            if self.SI:
                specific_heat *= (6.02214076e23  * 1.380649e-23)  # Convert to SI units (J/K)
                entropy *= (6.02214076e23 * 1.380649e-23)
                energy *= (6.02214076e23 * 1.380649e-23)
            results['energy'][i] = energy
            results['specific_heat'][i] = specific_heat 
            results['entropy'][i] = entropy
            
        return results
    
    def read_spin_expectations(self, spin_exp_dir):
        """
        Read spin expectation values from files in the specified directory.
        
        Args:
            spin_exp_dir: Directory containing spin expectation files
        
        Returns:
            Dictionary mapping temperature to spin expectation values
        """

        spin_expectations = {
            'sp': np.zeros_like(self.temp_values),
            'sm': np.zeros_like(self.temp_values),
            'sz': np.zeros_like(self.temp_values)
        }
        # Find all spin expectation files
        pattern = os.path.join(spin_exp_dir, "spin_expectations_T*.dat")
        spin_files = glob.glob(pattern)
        
        if not spin_files:
            print(f"No spin expectation files found in {spin_exp_dir}")
            return
            
        print(f"Found {len(spin_files)} spin expectation files")
        
        # Process each file
        for file_path in spin_files:
            # Extract temperature from filename
            match = re.search(r'T(\d+\.\d+)\.dat', file_path)
            if not match:
                print(f"Could not extract temperature from filename: {file_path}")
                continue
                
            temperature = float(match.group(1))
            print(f"Processing spin expectations for T = {temperature}")
            
            # Read file data
            try:
                data = np.loadtxt(file_path, skiprows=1)
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                continue
            
            # Extract site indices and spin expectations
            sites = data[:, 0].astype(int)
            sp_real = np.mean(data[:, 1])
            sp_imag = np.mean(data[:, 2])
            sm_real = np.mean(data[:, 3])
            sm_imag = np.mean(data[:, 4])
            sz_real = np.mean(data[:, 5])
            sz_imag = np.mean(data[:, 6])

            # Find the index corresponding to the temperature
            temp_index = np.argmin(np.abs(self.temp_values - temperature))
            if temp_index >= len(spin_expectations['sp']):
                print(f"Warning: Temperature index {temp_index} out of bounds for spin expectations")
                continue
            # Store the spin expectation values
            
            # Store as complex numbers in a dictionary
            spin_expectations['sp'][temp_index] = sp_real + 1j * sp_imag
            spin_expectations['sm'][temp_index] = sm_real + 1j * sm_imag
            spin_expectations['sz'][temp_index] = sz_real + 1j * sz_imag
            
        return spin_expectations


    def read_subcluster_info(self):
        """Read subcluster information from the provided file."""

        self.subcluster_info = {}
        
        subcluster_file = 'subclusters_info.txt'  # Default subcluster info file

        # Check if file exists in the specified directory
        filepath = os.path.join(self.cluster_dir, subcluster_file)
        print(f"Looking for subcluster info file at: {filepath}")
        if not os.path.exists(filepath):
            filepath = subcluster_file  # Try relative path
            if not os.path.exists(filepath):
                print(f"Warning: Subcluster info file not found at {subcluster_file}")
                return
                
        with open(filepath, 'r') as f:
            current_cluster = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith('Cluster'):
                    # Parse cluster header: "Cluster X (Order Y, L(c) = ...):" or "Cluster X (Order Y):"
                    match = re.match(r'Cluster (\d+) \(Order (\d+)', line)
                    if match:
                        current_cluster = int(match.group(1))
                        self.subcluster_info[current_cluster] = {'subclusters': {}}
                        
                elif 'No subclusters' in line:
                    # This cluster has no subclusters
                    continue
                        
                elif 'Subclusters:' in line:
                    if current_cluster is None:
                        print(f"Warning: Found Subclusters line without cluster header: {line}")
                        continue
                    # Parse subclusters: "(1, 2), (3, 4), ..."
                    subclusters_str = line.split('Subclusters:')[-1].strip()
                    if not subclusters_str:
                        continue
                        
                    # Extract all (id, multiplicity) pairs
                    pairs = re.findall(r'\((\d+),\s*(\d+)\)', subclusters_str)
                    for subcluster_id, multiplicity in pairs:
                        print(f"Adding subcluster {subcluster_id} with multiplicity {multiplicity} to cluster {current_cluster}")
                        self.subcluster_info[current_cluster]['subclusters'][int(subcluster_id)] = int(multiplicity)
    
    def get_subclusters(self, cluster_id):
        """
        Get all subclusters of a given cluster with their multiplicities.
        
        Returns:
            Dictionary mapping subcluster_id to its multiplicity in this cluster
        """
        # Try to use subcluster info if available
        if hasattr(self, 'subcluster_info') and cluster_id in self.subcluster_info:
            return self.subcluster_info[cluster_id]['subclusters']
        
        # Fallback to simple heuristic based on order
        # NOTE: This is a rough approximation and may miss same-order subclusters
        # For accurate NLCE, always use explicit subcluster info from subclusters_info.txt
        subclusters = {}
        order = self.clusters[cluster_id]['order']
        for cid, data in self.clusters.items():
            # Include all clusters of strictly lower order
            # Same-order subclusters require explicit subcluster info
            if data['order'] < order:
                subclusters[cid] = 1  # Assume multiplicity 1 as a fallback
                
        return subclusters
    
    def _topological_sort_clusters(self):
        """
        Sort clusters in dependency order using topological sort.
        
        A cluster must be processed AFTER all its subclusters, regardless of order.
        This handles cases where a cluster of order N contains subclusters of the same order N.
        
        Returns:
            List of (cluster_id, order) tuples in processing order
        """
        # Build dependency graph
        # deps[cluster_id] = set of subcluster IDs that must be processed first
        deps = {}
        for cluster_id in self.clusters:
            subclusters = self.get_subclusters(cluster_id)
            deps[cluster_id] = set(subclusters.keys())
        
        # Kahn's algorithm for topological sort
        # Start with clusters that have no dependencies (order 1 clusters)
        in_degree = {cid: len(d) for cid, d in deps.items()}
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        sorted_clusters = []
        
        while queue:
            # Sort queue by order (prefer lower order first for consistency)
            queue.sort(key=lambda cid: (self.clusters[cid]['order'], cid))
            cluster_id = queue.pop(0)
            order = self.clusters[cluster_id]['order']
            sorted_clusters.append((cluster_id, order))
            
            # Remove this cluster from all dependents' requirements
            for cid, dep_set in deps.items():
                if cluster_id in dep_set:
                    dep_set.remove(cluster_id)
                    in_degree[cid] -= 1
                    if in_degree[cid] == 0 and cid not in [x[0] for x in sorted_clusters] and cid not in queue:
                        queue.append(cid)
        
        # Check for cycles (shouldn't happen in valid NLCE data)
        if len(sorted_clusters) != len(self.clusters):
            remaining = set(self.clusters.keys()) - set(x[0] for x in sorted_clusters)
            print(f"WARNING: Dependency cycle detected! Remaining clusters: {remaining}")
            print("Falling back to order-based sort for remaining clusters.")
            # Add remaining clusters sorted by order
            for cid in sorted(remaining, key=lambda c: (self.clusters[c]['order'], c)):
                sorted_clusters.append((cid, self.clusters[cid]['order']))
        
        return sorted_clusters
    

    def print_subcluster_info(self):
        """Print out the subcluster information for all clusters."""
        # Read subcluster information if not already loaded
        if not hasattr(self, 'subcluster_info'):
            self.read_subcluster_info()
            
        if not hasattr(self, 'subcluster_info') or not self.subcluster_info:
            print("No subcluster information available.")
            return
            
        print("Subcluster Information:")
        print("----------------------")
        
        for cluster_id, info in sorted(self.subcluster_info.items()):
            print(f"Cluster {cluster_id}:")
            if 'subclusters' in info and info['subclusters']:
                print("  Subclusters (ID, Multiplicity):")
                for subcluster_id, multiplicity in sorted(info['subclusters'].items()):
                    print(f"    {subcluster_id}: {multiplicity}")
            else:
                print("  No subclusters")
            print()
    
    def calculate_weights(self):
        """Calculate weights for all clusters using the NLC principle."""
        # Read subcluster information if not already loaded
        if not hasattr(self, 'subcluster_info'):
            self.read_subcluster_info()

        self.print_subcluster_info()
            
        # Sort clusters in dependency order (handles same-order subclusters)
        sorted_clusters = self._topological_sort_clusters()
        
        print(f"\nProcessing {len(sorted_clusters)} clusters in dependency order")
        
        # Initialize weights dictionary
        if self.measure_spin:
            self.weights = {
                'energy': {},
                'specific_heat': {},
                'entropy': {},
                'sp': {},
                'sm': {},
                'sz': {}
            }
        else:
            self.weights = {
                'energy': {},
                'specific_heat': {},
                'entropy': {}   
            }
        
        # Track which clusters have valid weights (all required subclusters available)
        self.valid_weights = set()
        
        # Calculate weights for each cluster
        for cluster_id, order in sorted_clusters:
            if self.clusters[cluster_id]['eigenvalues'] is None:
                print(f"  Cluster {cluster_id} (order {order}): SKIPPED (no eigenvalues)")
                continue
            
            # Get subclusters with their multiplicities
            subclusters = self.get_subclusters(cluster_id)
            
            # Check if ALL required subclusters have valid weights
            # This is CRITICAL for correct recursive weight calculation
            missing_subclusters = []
            for sub_id in subclusters.keys():
                if sub_id not in self.valid_weights:
                    missing_subclusters.append(sub_id)
            
            if missing_subclusters:
                print(f"  Cluster {cluster_id} (order {order}): SKIPPED - missing weights for subclusters {missing_subclusters}")
                print(f"    Cannot compute weight because W(c) = P(c) - Σ Y_cs × W(s) requires ALL subcluster weights")
                continue
                
            # Get thermodynamic quantities for this cluster
            quantities = self.calculate_thermodynamic_quantities(
                self.clusters[cluster_id]['eigenvalues'],
            )
            
            # Calculate weights for energy and specific heat
            for prop in ['energy', 'specific_heat', 'entropy']:
                # Property of the cluster
                property_value = quantities[prop]
                # Subtract contributions from all subclusters with correct multiplicities
                for subcluster_id, multiplicity in subclusters.items():
                    if subcluster_id in self.weights[prop]:
                        property_value -= self.weights[prop][subcluster_id] * multiplicity
                # Store the weight
                self.weights[prop][cluster_id] = property_value

            # Mark this cluster as having valid weights for thermodynamic properties
            self.valid_weights.add(cluster_id)


            if self.measure_spin:
                # Read spin expectation values if needed
                spin_exp_dir = os.path.join(self.eigenvalue_dir, f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}/output/spin_expectations")
                quantities_spin =  self.read_spin_expectations(spin_exp_dir)
                
                print(f"Spin expectation values for cluster {cluster_id}: {quantities_spin}")

                for prop in ['sp', 'sm', 'sz']:
                    # Get the spin expectation value
                    spin_value = quantities_spin[prop]
                    # Subtract contributions from all subclusters with correct multiplicities
                    for subcluster_id, multiplicity in subclusters.items():
                        if subcluster_id in self.weights[prop]:
                            spin_value -= self.weights[prop][subcluster_id] * multiplicity
                    # Store the weight
                    self.weights[prop][cluster_id] = spin_value
            



    def euler_resummation(self, partial_sums, l=3):
        """
        Apply Euler transformation for series acceleration.
        
        Implementation follows Tang–Khatami–Rigol (arXiv:1207.3366):
        Keep first (n-l) bare terms, apply Euler transform to the tail using
        forward differences with 2^{-(k+1)} weights.
        
        Args:
            partial_sums: Array of partial sums S_n = sum_{k=0}^n a_k
            l: Number of highest-order terms to keep "as is" (default: 3)
            
        Returns:
            Accelerated series approximation
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else 0.0
        
        # Convert partial sums to increments (the a_n terms)
        increments = [partial_sums[0]]
        for i in range(1, n):
            increments.append(partial_sums[i] - partial_sums[i-1])
        
        # Choose l adaptively if sequence is short
        l_use = min(l, max(2, n - 3))
        
        if n <= l_use:
            # Not enough terms for Euler, return last partial sum
            return partial_sums[-1]
        
        # Keep first (n - l_use) bare terms as-is
        bare_sum = partial_sums[n - l_use - 1] if n - l_use - 1 >= 0 else (
            np.zeros_like(partial_sums[-1]) if len(partial_sums.shape) > 1 
            else 0.0
        )
        
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
        euler_tail = np.zeros_like(partial_sums[-1]) if len(partial_sums.shape) > 1 else 0.0
        for k, diff_row in enumerate(diff_triangle):
            if len(diff_row) > 0:
                euler_tail += diff_row[0] / (2**(k+1))
        
        return bare_sum + euler_tail
    
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
            return sequence[-1] if n > 0 else 0.0
            
        # Initialize epsilon table
        if len(sequence.shape) > 1:
            eps = np.zeros((n + 1, n + 1, sequence.shape[1]))
        else:
            eps = np.zeros((n + 1, n + 1))
            
        # Set initial values
        eps[0, :] = 0.0
        for i in range(n):
            eps[1, i] = sequence[i]
            
        # Fill epsilon table
        for k in range(2, n + 1):
            for i in range(n - k + 1):
                denominator = eps[k-1, i+1] - eps[k-1, i]
                # Avoid division by zero
                if np.any(np.abs(denominator) < 1e-15):
                    eps[k, i] = eps[k-1, i+1]
                else:
                    eps[k, i] = eps[k-2, i+1] + 1.0 / denominator
                    
        # Return the most accelerated value (highest even k)
        max_even_k = 2 * ((n) // 2)
        if max_even_k > 0:
            return eps[max_even_k, 0]
        else:
            return sequence[-1]
    
    def pade_approximant(self, coefficients, m=None, n=None):
        """
        Construct Padé approximant [m/n] from series coefficients.
        
        Args:
            coefficients: Series coefficients a_k
            m: Degree of numerator (default: len(coeffs)//2)
            n: Degree of denominator (default: len(coeffs)//2)
            
        Returns:
            Function representing the Padé approximant
        """
        L = len(coefficients)
        if m is None:
            m = L // 2
        if n is None:
            n = L - m - 1
            
        if m + n + 1 > L:
            m = L // 2
            n = L - m - 1
            
        # Set up linear system for denominator coefficients
        # c_k + sum_{j=1}^n b_j * c_{k-j} = 0 for k = m+1, ..., m+n
        if n > 0:
            A = np.zeros((n, n))
            b_vec = np.zeros(n)
            
            for i in range(n):
                k = m + 1 + i
                if k < L:
                    b_vec[i] = -coefficients[k]
                    for j in range(n):
                        if k - 1 - j >= 0 and k - 1 - j < L:
                            A[i, j] = coefficients[k - 1 - j]
                            
            # Solve for denominator coefficients
            try:
                q_coeffs = np.linalg.solve(A, b_vec)
                q_coeffs = np.concatenate([[1.0], q_coeffs])
            except np.linalg.LinAlgError:
                q_coeffs = np.array([1.0])
                n = 0
        else:
            q_coeffs = np.array([1.0])
            
        # Calculate numerator coefficients
        p_coeffs = np.zeros(m + 1)
        for k in range(min(m + 1, L)):
            p_coeffs[k] = coefficients[k]
            for j in range(1, min(k + 1, len(q_coeffs))):
                if k - j >= 0 and k - j < L:
                    p_coeffs[k] += q_coeffs[j] * coefficients[k - j]
                    
        return p_coeffs, q_coeffs
    
    def evaluate_pade(self, p_coeffs, q_coeffs, x):
        """Evaluate Padé approximant at point x."""
        p_val = np.polyval(p_coeffs[::-1], x)
        q_val = np.polyval(q_coeffs[::-1], x)
        
        # Avoid division by zero
        mask = np.abs(q_val) > 1e-15
        result = np.zeros_like(x)
        result[mask] = p_val[mask] / q_val[mask]
        result[~mask] = p_val[~mask]  # Use polynomial approximation when denominator is small
        
        return result
    
    def shanks_transform(self, sequence):
        """
        Apply Shanks transformation for series acceleration.
        
        Args:
            sequence: Array of partial sums
            
        Returns:
            Accelerated approximation
        """
        n = len(sequence)
        if n < 3:
            return sequence[-1] if n > 0 else 0.0
            
        # Shanks transformation: S' = (S_{n+1}*S_{n-1} - S_n^2) / (S_{n+1} - 2*S_n + S_{n-1})
        s_prev = sequence[-3]
        s_curr = sequence[-2] 
        s_next = sequence[-1]
        
        denominator = s_next - 2*s_curr + s_prev
        
        # Avoid division by zero
        if np.any(np.abs(denominator) < 1e-15):
            return s_next
        else:
            return (s_next * s_prev - s_curr**2) / denominator
    
    def aitken_delta2(self, sequence):
        """
        Apply Aitken's Δ² method for series acceleration.
        
        Args:
            sequence: Array of partial sums
            
        Returns:
            Accelerated approximation
        """
        return self.shanks_transform(sequence)  # Aitken's Δ² is equivalent to Shanks
    
    def analyze_convergence(self, partial_sums):
        """
        Analyze convergence properties of the series.
        
        Args:
            partial_sums: Array of partial sums
            
        Returns:
            Dictionary with convergence metrics
        """
        n = len(partial_sums)
        if n < 3:
            return {'converged': False, 'oscillatory': False, 'ratio_test': None}
            
        # Calculate differences
        diffs = np.diff(partial_sums, axis=0)
        
        # Check for convergence (differences getting smaller)
        if n > 3:
            recent_diffs = np.abs(diffs[-3:])
            converged = np.all(recent_diffs[-1] <= recent_diffs[-2]) and np.all(recent_diffs[-2] <= recent_diffs[-3])
        else:
            converged = False
            
        # Check for oscillatory behavior
        if n > 4:
            signs = np.sign(diffs[-4:])
            oscillatory = np.all(signs[1:] != signs[:-1])  # Alternating signs
        else:
            oscillatory = False
            
        # Ratio test for convergence
        if n > 2:
            ratios = np.abs(diffs[1:] / diffs[:-1])
            avg_ratio = np.mean(ratios[-min(3, len(ratios)):])  # Average of last few ratios
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
            partial_sums: Array of partial sums for analysis
            
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
            
        # For small number of terms, use Shanks
        if n <= 5:
            return 'shanks'

        # For slowly converging series, try Wynn's epsilon
        if convergence['ratio_test'] is not None and convergence['ratio_test'] > 0.5:
            if n >= 5:  # Wynn needs at least 5 terms for stability
                return 'wynn'
            else:
                return 'euler'
                

            
        # Default to Wynn's epsilon for general case
        return 'wynn'
    
    def apply_resummation(self, partial_sums, method='auto'):
        """
        Apply the specified resummation method.
        
        Args:
            partial_sums: Array of partial sums
            method: Resummation method ('auto', 'direct', 'euler', 'wynn', 'shanks', 'pade')
            
        Returns:
            Accelerated series approximation
        """
        if len(partial_sums) == 0:
            return np.zeros_like(self.temp_values)
            
        if method == 'auto':
            method = self.select_resummation_method(partial_sums)
            
        print(f"Using resummation method: {method}")
        
        if method == 'direct':
            return partial_sums[-1]
        elif method == 'euler':
            return self.euler_resummation(partial_sums)
        elif method == 'wynn':
            return self.wynn_epsilon(partial_sums)
        elif method == 'shanks':
            return self.shanks_transform(partial_sums)
        elif method == 'aitken':
            return self.aitken_delta2(partial_sums)
        elif method == 'pade':
            # For Padé, we need series coefficients, not partial sums
            # Convert partial sums to coefficients
            coeffs = np.diff(np.concatenate([[np.zeros_like(partial_sums[0])], partial_sums]), axis=0)
            p_coeffs, q_coeffs = self.pade_approximant(coeffs)
            # Evaluate at x=1 (summing the series)
            return self.evaluate_pade(p_coeffs, q_coeffs, np.ones_like(self.temp_values))
        else:
            print(f"Unknown method {method}, using direct summation")
            return partial_sums[-1]
    
    def sum_nlc(self, resummation_method='auto', order_cutoff=None):
        """
        Perform the NLC summation with automatic resummation method selection.
        
        Args:
            resummation_method: Method for series acceleration ('auto', 'direct', 'euler', 'wynn', 'shanks', 'pade')
            order_cutoff: Maximum order to include in the summation
            
        Returns:
            Dictionary with summed properties
        """

        if self.measure_spin:
            # Initialize results for spin expectation values
            results = {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
                'entropy': np.zeros_like(self.temp_values),
                'sp': np.zeros_like(self.temp_values),
                'sm': np.zeros_like(self.temp_values),
                'sz': np.zeros_like(self.temp_values)
            }
            properties = ['energy', 'specific_heat', 'entropy', 'sp', 'sm', 'sz']
        else:
            results = {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
                'entropy': np.zeros_like(self.temp_values)
            }
            properties = ['energy', 'specific_heat', 'entropy']
    
        # Calculate the NLC sum for each property
        for prop in properties:
            if prop not in self.weights:
                continue
                
            print(f"Processing property: {prop}")
            
            # Sum by order
            sum_by_order = defaultdict(lambda: np.zeros_like(self.temp_values))
            
            for cluster_id, weight in self.weights[prop].items():
                order = self.clusters[cluster_id]['order']
                if order_cutoff is not None and order > order_cutoff:
                    continue
                    
                sum_by_order[order] += weight * self.clusters[cluster_id]['multiplicity']
            
            # Create array of partial sums
            if sum_by_order:
                max_order = max(sum_by_order.keys())
                partial_sums = []
                cumulative_sum = np.zeros_like(self.temp_values)
                
                for order in range(max_order + 1):
                    if order in sum_by_order:
                        cumulative_sum += sum_by_order[order]
                    partial_sums.append(cumulative_sum.copy())
                
                partial_sums = np.array(partial_sums)
                
                # Apply resummation
                results[prop] = self.apply_resummation(partial_sums, method=resummation_method)
            else:
                print(f"Warning: No weights found for property {prop}")
                results[prop] = np.zeros_like(self.temp_values)
        
        # Special handling for specific heat - can also compute as derivative of energy
        if 'energy' in results:
            # Calculate specific heat as derivative of energy
            energy_derivative = -np.gradient(results['energy'], self.temp_values) / (self.temp_values**2)
            
            # Use the maximum of the two methods to avoid negative specific heat
            if 'specific_heat' in results:
                results['specific_heat'] = np.maximum(results['specific_heat'], energy_derivative)
            else:
                results['specific_heat'] = np.maximum(energy_derivative, 0.0)
        
        return results
    
    def run(self, resummation_method='auto', order_cutoff=None):
        """Run the full NLC calculation."""
        print("Reading cluster information...")
        self.read_clusters()
        
        print("Reading eigenvalues...")
        self.read_eigenvalues()
        
        print("Calculating weights...")
        self.calculate_weights()
        
        print("Performing NLC summation...")
        results = self.sum_nlc(resummation_method, order_cutoff)

        return results
    
    def compare_resummation_methods(self, order_cutoff=None, save_comparison=False):
        """
        Compare different resummation methods and provide diagnostic information.
        
        Args:
            order_cutoff: Maximum order to include in summation
            save_comparison: If True, save comparison plots and data
            
        Returns:
            Dictionary with results from different methods and diagnostics
        """
        methods = ['direct', 'euler', 'wynn', 'shanks', 'aitken']
        comparison_results = {}
        
        print("Comparing resummation methods...")
        
        # Get partial sums for analysis
        prop = 'energy'  # Use energy as representative property
        if prop not in self.weights:
            print("No energy weights available for comparison")
            return {}
            
        sum_by_order = defaultdict(lambda: np.zeros_like(self.temp_values))
        
        for cluster_id, weight in self.weights[prop].items():
            order = self.clusters[cluster_id]['order']
            if order_cutoff is not None and order > order_cutoff:
                continue
            sum_by_order[order] += weight * self.clusters[cluster_id]['multiplicity']
        
        if not sum_by_order:
            print("No data available for comparison")
            return {}
            
        # Create partial sums array
        max_order = max(sum_by_order.keys())
        partial_sums = []
        cumulative_sum = np.zeros_like(self.temp_values)
        
        for order in range(max_order + 1):
            if order in sum_by_order:
                cumulative_sum += sum_by_order[order]
            partial_sums.append(cumulative_sum.copy())
        
        partial_sums = np.array(partial_sums)
        
        # Analyze convergence
        convergence_info = self.analyze_convergence(partial_sums)
        comparison_results['convergence_analysis'] = convergence_info
        
        # Test each method
        for method in methods:
            try:
                result = self.apply_resummation(partial_sums, method=method)
                comparison_results[method] = {
                    'result': result,
                    'final_value_avg': np.mean(result),
                    'final_value_std': np.std(result)
                }
                
                # Calculate convergence metrics
                if len(partial_sums) > 1:
                    direct_result = partial_sums[-1]
                    improvement = np.mean(np.abs(result - direct_result))
                    comparison_results[method]['improvement_from_direct'] = improvement
                    
            except Exception as e:
                print(f"Error with method {method}: {e}")
                comparison_results[method] = {'error': str(e)}
        
        # Automatic selection
        auto_method = self.select_resummation_method(partial_sums)
        comparison_results['auto_selected'] = auto_method
        
        # Print summary
        print("\nResummation Method Comparison:")
        print("=" * 50)
        print(f"Convergence Analysis:")
        print(f"  Converged: {convergence_info['converged']}")
        print(f"  Oscillatory: {convergence_info['oscillatory']}")
        print(f"  Ratio test: {convergence_info['ratio_test']:.4f}" if convergence_info['ratio_test'] else "  Ratio test: N/A")
        print(f"  Auto-selected method: {auto_method}")
        print("\nMethod Results (average final values):")
        
        for method in methods:
            if method in comparison_results and 'result' in comparison_results[method]:
                avg_val = comparison_results[method]['final_value_avg']
                print(f"  {method:10s}: {avg_val:.6e}")
        
        if save_comparison:
            self._save_comparison_plots(comparison_results, partial_sums)
            
        return comparison_results
    
    def _save_comparison_plots(self, comparison_results, partial_sums):
        """Save comparison plots for different resummation methods."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Convergence of partial sums
            orders = np.arange(len(partial_sums))
            temp_mid = len(self.temp_values) // 2  # Use middle temperature for visualization
            
            ax1.plot(orders, partial_sums[:, temp_mid], 'bo-', label='Partial sums')
            ax1.set_xlabel('Order')
            ax1.set_ylabel('Energy (middle T)')
            ax1.set_title('Convergence of Partial Sums')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Method comparison at middle temperature
            methods = ['direct', 'euler', 'wynn', 'shanks', 'aitken']
            values = []
            labels = []
            
            for method in methods:
                if method in comparison_results and 'result' in comparison_results[method]:
                    values.append(comparison_results[method]['result'][temp_mid])
                    labels.append(method)
            
            if values:
                ax2.bar(labels, values)
                ax2.set_ylabel('Energy (middle T)')
                ax2.set_title('Method Comparison')
                ax2.tick_params(axis='x', rotation=45)
            
            # Plot 3: Temperature dependence comparison
            temp_range = self.temp_values
            for method in methods:
                if method in comparison_results and 'result' in comparison_results[method]:
                    result = comparison_results[method]['result']
                    ax3.plot(temp_range, result, label=method, alpha=0.7)
            
            ax3.set_xlabel('Temperature')
            ax3.set_ylabel('Energy')
            ax3.set_title('Temperature Dependence')
            ax3.set_xscale('log')
            ax3.legend()
            ax3.grid(True)
            
            # Plot 4: Convergence analysis
            if len(partial_sums) > 1:
                diffs = np.abs(np.diff(partial_sums[:, temp_mid]))
                ax4.semilogy(orders[1:], diffs, 'ro-')
                ax4.set_xlabel('Order')
                ax4.set_ylabel('|Difference|')
                ax4.set_title('Convergence Rate')
                ax4.grid(True)
            
            plt.tight_layout()
            plt.savefig('resummation_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("Comparison plots saved to 'resummation_comparison.png'")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")
        except Exception as e:
            print(f"Error creating plots: {e}")
    
    def plot_results(self, results, save_path=None):
        """Plot energy and specific heat vs temperature."""
        temperatures = 1.0 / self.temp_values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot energy
        ax1.plot(temperatures, results['energy'], 'b-')
        ax1.set_xlabel('Temperature (T)')
        ax1.set_ylabel('Energy per site')
        ax1.set_title('Energy vs Temperature')
        ax1.set_xscale('log')
        
        # Plot specific heat
        ax2.plot(temperatures, results['specific_heat'], 'r-')
        ax2.set_xlabel('Temperature (T)')
        ax2.set_ylabel('Specific Heat per site')
        ax2.set_title('Specific Heat vs Temperature')
        ax2.set_xscale('log')

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NLC calculation for lattice models')
    parser.add_argument('--cluster_dir', required=True, help='Directory containing cluster information files')
    parser.add_argument('--eigenvalue_dir', required=True, help='Directory containing eigenvalue files from ED calculations')
    parser.add_argument('--output_dir', default='.', help='Directory to save output files')
    parser.add_argument('--resummation_method', default='auto', 
                       choices=['auto', 'direct', 'euler', 'wynn', 'shanks', 'aitken', 'pade'],
                       help='Resummation method for series acceleration')
    parser.add_argument('--euler_resum', action='store_true', help='Use Euler resummation (deprecated, use --resummation_method euler)')
    parser.add_argument('--order_cutoff', type=int, help='Maximum order to include in summation')
    parser.add_argument('--plot', action='store_true', help='Generate plot of results')
    parser.add_argument('--SI_units', action='store_true', help='Use SI units for output')
    parser.add_argument('--temp_min', type=float, default=1e-4, help='Minimum temperature for calculations')
    parser.add_argument('--temp_max', type=float, default=1.0, help='Maximum temperature for calculations')
    parser.add_argument('--temp_bins', type=int, default=200, help='Number of temperature points to calculate')
    parser.add_argument('--measure_spin', action='store_true', help='Measure spin expectation values')
    parser.add_argument('--compare_methods', action='store_true', help='Compare different resummation methods')
    parser.add_argument('--save_comparison', action='store_true', help='Save comparison plots and data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create NLC instance
    nlc = NLCExpansion(args.cluster_dir, args.eigenvalue_dir, args.temp_min, args.temp_max, args.temp_bins, args.measure_spin, args.SI_units)
    
    # Handle backward compatibility for euler_resum flag
    resummation_method = args.resummation_method
    if args.euler_resum and resummation_method == 'auto':
        resummation_method = 'euler'
        print("Note: --euler_resum flag is deprecated, use --resummation_method euler instead")
    
    # Run NLC calculation
    results = nlc.run(resummation_method=resummation_method, order_cutoff=args.order_cutoff)
    
    # Compare methods if requested
    if args.compare_methods:
        print("\n" + "="*60)
        print("COMPARING RESUMMATION METHODS")
        print("="*60)
        comparison = nlc.compare_resummation_methods(
            order_cutoff=args.order_cutoff,
            save_comparison=args.save_comparison
        )
    
    # Save results in separate files for each quantity
    energy_file = os.path.join(args.output_dir, "nlc_energy.txt")
    specific_heat_file = os.path.join(args.output_dir, "nlc_specific_heat.txt")
    entropy_file = os.path.join(args.output_dir, "nlc_entropy.txt")

    # Save energy data
    with open(energy_file, 'w') as f:
        f.write("# Temperature\tEnergy\n")
        for i, temp in enumerate(nlc.temp_values):
            f.write(f"{temp:.8e}\t{results['energy'][i]:.8e}\n")

    # Save specific heat data
    with open(specific_heat_file, 'w') as f:
        f.write("# Temperature\tSpecific_Heat\n")
        for i, temp in enumerate(nlc.temp_values):
            f.write(f"{temp:.8e}\t{results['specific_heat'][i]:.8e}\n")

    # Save entropy data
    with open(entropy_file, 'w') as f:
        f.write("# Temperature\tEntropy\n")
        for i, temp in enumerate(nlc.temp_values):
            f.write(f"{temp:.8e}\t{results['entropy'][i]:.8e}\n")

    # Save spin expectation values if requested
    if args.measure_spin:
        spin_file = os.path.join(args.output_dir, "nlc_spin_expectations.txt")
        with open(spin_file, 'w') as f:
            f.write("# Temperature\tsp\tsp_imag\tsm\tsm_imag\tsz\tsz_imag\n")
            for i, temp in enumerate(nlc.temp_values):
                sp = results['sp'][i]
                sm = results['sm'][i]
                sz = results['sz'][i]
                f.write(f"{temp:.8e}\t{sp.real:.8e}\t{sp.imag:.8e}\t{sm.real:.8e}\t{sm.imag:.8e}\t{sz.real:.8e}\t{sz.imag:.8e}\n")

    # Plot results if requested
    if args.plot:
        temperatures = nlc.temp_values
        
        # Plot energy
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['energy'], 'b-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Energy per site')
        plt.title('Energy vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        energy_plot_path = os.path.join(args.output_dir, "nlc_energy.png")
        plt.savefig(energy_plot_path)
        plt.close()
        
        # Plot specific heat
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['specific_heat'], 'r-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Specific Heat per site')
        plt.title('Specific Heat vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        specific_heat_plot_path = os.path.join(args.output_dir, "nlc_specific_heat.png")
        plt.savefig(specific_heat_plot_path)
        plt.close()
        
        # Plot entropy
        plt.figure(figsize=(8, 6))
        plt.plot(temperatures, results['entropy'], 'g-')
        plt.xlabel('Temperature (T)')
        plt.ylabel('Entropy per site')
        plt.title('Entropy vs Temperature')
        plt.xscale('log')
        plt.tight_layout()
        entropy_plot_path = os.path.join(args.output_dir, "nlc_entropy.png")
        plt.savefig(entropy_plot_path)
        plt.close()

        # Plot spin expectation values if requested
        if args.measure_spin:
            plt.figure(figsize=(8, 6))
            plt.plot(temperatures, results['sp'], 'm-', label='sp')
            plt.plot(temperatures, results['sm'], 'c-', label='sm')
            plt.plot(temperatures, results['sz'], 'y-', label='sz')
            plt.xlabel('Temperature (T)')
            plt.ylabel('Spin Expectation Values')
            plt.title('Spin Expectation Values vs Temperature')
            plt.xscale('log')
            plt.legend()
            plt.tight_layout()
            spin_plot_path = os.path.join(args.output_dir, "nlc_spin_expectations.png")
            plt.savefig(spin_plot_path)
            plt.close()
    
    print(f"NLC calculation completed! Results saved to {args.output_dir}")
    