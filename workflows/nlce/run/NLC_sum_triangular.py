#!/usr/bin/env python3
"""
NLC (Numerical Linked Cluster Expansion) summation utility for triangular lattice.
Calculates thermodynamic properties of a triangular lattice using cluster expansion.

The triangular lattice has 2 triangles per site (up and down), so the normalization
differs from pyrochlore (which has 2 sites per tetrahedron).
"""

import os
import numpy as np
import glob
import re
from collections import defaultdict
from scipy.optimize import curve_fit
import argparse
import matplotlib.pyplot as plt

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. HDF5 file reading will not be available.")


class NLCExpansionTriangular:
    """NLCE calculator for triangular lattice."""
    
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min=None, temp_max=None, num_temps=None, 
                 measure_spin=False, SI_units=False, temp_points=None):
        """
        Initialize the NLC expansion calculator for triangular lattice.
        
        All quantities are in Kelvin:
            - Hamiltonian couplings (and hence eigenvalues) must be in Kelvin
            - Temperatures are in Kelvin
            - Energy output is in Kelvin (or J/mol if SI)
        
        Temperature grid can be specified in two ways:
            1. temp_min/temp_max/num_temps: logarithmic grid (legacy)
            2. temp_points: explicit array of temperature values (preferred for fitting)
        If temp_points is provided, temp_min/temp_max/num_temps are ignored.
        
        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_min: Minimum temperature (Kelvin) — used only if temp_points is None
            temp_max: Maximum temperature (Kelvin) — used only if temp_points is None
            num_temps: Number of temperature points — used only if temp_points is None
            measure_spin: Whether to compute spin expectation values
            SI_units: Convert to SI units (J/(mol·K) for C and S, J/mol for E)
            temp_points: Explicit temperature array (Kelvin). Overrides temp_min/max/num.
                      
        SI Unit Conversion:
            - Specific heat: C_SI = R × C where R = 8.314 J/(mol·K)
            - Entropy: S_SI = R × S  
            - Energy: E_SI = R × E [J/mol] (E in Kelvin)
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir
        
        self.SI = SI_units
        self.measure_spin = measure_spin
        
        if temp_points is not None:
            self.temp_values = np.sort(np.asarray(temp_points, dtype=float))
        else:
            if temp_min is None or temp_max is None or num_temps is None:
                raise ValueError("Must provide either temp_points or all of temp_min/temp_max/num_temps")
            self.temp_values = np.logspace(np.log10(temp_min), np.log10(temp_max), num_temps)
        
        self.clusters = {}
        self.weights = {}
        self.subcluster_info = {}
        self.valid_weights = set()
        
    def read_clusters(self):
        """Read all cluster information from files in the cluster directory."""
        pattern = os.path.join(self.cluster_dir, "cluster_*_order_*.dat")
        cluster_files = glob.glob(pattern)
        
        for file_path in cluster_files:
            match = re.search(r'cluster_(\d+)_order_(\d+)', file_path)
            if not match:
                continue
                
            cluster_id, order = int(match.group(1)), int(match.group(2))
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            multiplicity = None
            num_vertices = None
            
            for line in lines:
                if line.startswith("# Multiplicity") and ":" in line:
                    mult_str = line.split(":")[-1].strip()
                    # Handle formats like "1/3 = 0.333333" or just "0.333333"
                    if "=" in mult_str:
                        mult_str = mult_str.split("=")[-1].strip()
                    # Handle fractional format like "1/3"
                    if "/" in mult_str:
                        parts = mult_str.split("/")
                        multiplicity = float(parts[0]) / float(parts[1])
                    else:
                        multiplicity = float(mult_str)
                elif line.startswith("# Number of vertices:"):
                    num_vertices = int(line.split(":")[1].strip())
                
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
                'eigenvalues': None,
            }
    
    def read_eigenvalues(self):
        """Read eigenvalues for each cluster from ED output files."""
        for cluster_id in self.clusters:
            cluster_base_dir = os.path.join(
                self.eigenvalue_dir, 
                f"cluster_{cluster_id}_order_{self.clusters[cluster_id]['order']}"
            )
            cluster_output_dir = os.path.join(cluster_base_dir, "output")
            
            # Try HDF5 file first
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
            
            # Fall back to text file
            eigenvalue_file = os.path.join(cluster_output_dir, "eigenvalues.txt")
            if os.path.exists(eigenvalue_file):
                with open(eigenvalue_file, 'r') as f:
                    eigenvalues = [float(line.strip()) for line in f if line.strip()]
                self.clusters[cluster_id]['eigenvalues'] = np.array(eigenvalues)
                continue
            
            print(f"Warning: Eigenvalue data not found for cluster {cluster_id}")
    
    def read_subcluster_info(self):
        """Read subcluster information from the provided file."""
        self.subcluster_info = {}
        
        filepath = os.path.join(self.cluster_dir, 'subclusters_info.txt')
        if not os.path.exists(filepath):
            print(f"Warning: Subcluster info file not found at {filepath}")
            return
                
        with open(filepath, 'r') as f:
            current_cluster = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith('Cluster'):
                    match = re.match(r'Cluster (\d+) \(Order (\d+)', line)
                    if match:
                        current_cluster = int(match.group(1))
                        self.subcluster_info[current_cluster] = {'subclusters': {}}
                        
                elif 'No subclusters' in line:
                    continue
                        
                elif 'Subclusters:' in line:
                    if current_cluster is None:
                        continue
                    subclusters_str = line.split('Subclusters:')[-1].strip()
                    if not subclusters_str:
                        continue
                        
                    pairs = re.findall(r'\((\d+),\s*(\d+)\)', subclusters_str)
                    for subcluster_id, multiplicity in pairs:
                        self.subcluster_info[current_cluster]['subclusters'][int(subcluster_id)] = int(multiplicity)
    
    def get_subclusters(self, cluster_id):
        """Get all subclusters of a given cluster with their multiplicities."""
        if hasattr(self, 'subcluster_info') and cluster_id in self.subcluster_info:
            return self.subcluster_info[cluster_id]['subclusters']
        
        # Fallback to order-based heuristic
        # WARNING: This is mathematically wrong for most cluster topologies.
        # Correct subcluster embeddings should come from subclusters_info.txt.
        import warnings
        warnings.warn(
            f"Cluster {cluster_id}: using order-based subcluster heuristic (fallback). "
            f"This likely produces incorrect NLCE weights. "
            f"Ensure subclusters_info.txt is generated correctly.",
            RuntimeWarning
        )
        subclusters = {}
        order = self.clusters[cluster_id]['order']
        for cid, data in self.clusters.items():
            if data['order'] < order:
                subclusters[cid] = 1
        return subclusters
    
    def _topological_sort_clusters(self):
        """Sort clusters in dependency order using topological sort."""
        deps = {}
        for cluster_id in self.clusters:
            subclusters = self.get_subclusters(cluster_id)
            deps[cluster_id] = set(subclusters.keys())
        
        in_degree = {cid: len(d) for cid, d in deps.items()}
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        sorted_clusters = []
        
        while queue:
            queue.sort(key=lambda cid: (self.clusters[cid]['order'], cid))
            cluster_id = queue.pop(0)
            order = self.clusters[cluster_id]['order']
            sorted_clusters.append((cluster_id, order))
            
            for cid, dep_set in deps.items():
                if cluster_id in dep_set:
                    dep_set.remove(cluster_id)
                    in_degree[cid] -= 1
                    if in_degree[cid] == 0 and cid not in [x[0] for x in sorted_clusters] and cid not in queue:
                        queue.append(cid)
        
        if len(sorted_clusters) != len(self.clusters):
            remaining = set(self.clusters.keys()) - set(x[0] for x in sorted_clusters)
            for cid in sorted(remaining, key=lambda c: (self.clusters[c]['order'], c)):
                sorted_clusters.append((cid, self.clusters[cid]['order']))
        
        return sorted_clusters
    
    def calculate_thermodynamic_quantities(self, eigenvalues):
        """Calculate thermodynamic quantities from eigenvalues."""
        # Validate eigenvalues
        if eigenvalues is None or len(eigenvalues) == 0:
            print("  WARNING: Empty eigenvalue array, returning zeros")
            return {
                'energy': np.zeros_like(self.temp_values),
                'specific_heat': np.zeros_like(self.temp_values),
                'entropy': np.zeros_like(self.temp_values)
            }
        if not np.all(np.isfinite(eigenvalues)):
            print(f"  WARNING: {np.sum(~np.isfinite(eigenvalues))} non-finite eigenvalues detected, filtering")
            eigenvalues = eigenvalues[np.isfinite(eigenvalues)]
            if len(eigenvalues) == 0:
                return {
                    'energy': np.zeros_like(self.temp_values),
                    'specific_heat': np.zeros_like(self.temp_values),
                    'entropy': np.zeros_like(self.temp_values)
                }

        results = {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values)
        }
        
        for i, temp in enumerate(self.temp_values):
            if temp < 1e-10:
                ground_state_energy = np.min(eigenvalues)
                results['energy'][i] = ground_state_energy
                results['specific_heat'][i] = 0.0
                results['entropy'][i] = 0.0
                continue
                
            ground_state_energy = np.min(eigenvalues)
            shifted_eigenvalues = eigenvalues - ground_state_energy
            
            exp_terms = np.exp(-shifted_eigenvalues / temp)
            Z_shifted = np.sum(exp_terms)
            
            energy = np.sum(eigenvalues * exp_terms) / Z_shifted
            energy_squared = np.sum(eigenvalues**2 * exp_terms) / Z_shifted
            
            specific_heat = (energy_squared - energy**2) / (temp * temp)
            entropy = np.log(Z_shifted) + (energy - ground_state_energy) / temp
            
            if self.SI:
                # Gas constant R = NA * kB = 8.314462618 J/(mol·K)
                R = 6.02214076e23 * 1.380649e-23  # ≈ 8.314 J/(mol·K)
                # Specific heat per mole: C_SI [J/(mol·K)] = R × C [dimensionless]
                specific_heat *= R
                # Entropy per mole: S_SI [J/(mol·K)] = R × S [dimensionless]
                entropy *= R
                # Energy per mole: E_SI [J/mol] = R × E_K (E in Kelvin)
                energy *= R
                
            results['energy'][i] = energy
            results['specific_heat'][i] = specific_heat 
            results['entropy'][i] = entropy
            
        return results
    
    def calculate_weights(self):
        """Calculate weights for all clusters using the NLC principle."""
        if not hasattr(self, 'subcluster_info') or not self.subcluster_info:
            self.read_subcluster_info()
            
        sorted_clusters = self._topological_sort_clusters()
        
        print(f"\nProcessing {len(sorted_clusters)} clusters in dependency order")
        
        self.weights = {
            'energy': {},
            'specific_heat': {},
            'entropy': {}
        }
        
        self.valid_weights = set()
        
        for cluster_id, order in sorted_clusters:
            if self.clusters[cluster_id]['eigenvalues'] is None:
                print(f"  Cluster {cluster_id} (order {order}): SKIPPED (no eigenvalues)")
                continue
            
            subclusters = self.get_subclusters(cluster_id)
            
            missing_subclusters = []
            for sub_id in subclusters.keys():
                if sub_id not in self.valid_weights:
                    missing_subclusters.append(sub_id)
            
            if missing_subclusters:
                print(f"  Cluster {cluster_id} (order {order}): SKIPPED - missing weights for {missing_subclusters}")
                continue
                
            quantities = self.calculate_thermodynamic_quantities(
                self.clusters[cluster_id]['eigenvalues']
            )
            
            for prop in ['energy', 'specific_heat', 'entropy']:
                property_value = quantities[prop].copy()
                for subcluster_id, multiplicity in subclusters.items():
                    if subcluster_id in self.weights[prop]:
                        property_value -= self.weights[prop][subcluster_id] * multiplicity
                self.weights[prop][cluster_id] = property_value

            self.valid_weights.add(cluster_id)
            print(f"  Cluster {cluster_id} (order {order}): OK")
    
    def perform_summation(self, max_order=None):
        """Perform NLCE summation up to specified order.
        
        Includes order 0 (single-site seed) so that entropy gets the
        ln(2)-per-site baseline and energy gets the free-spin ground state.
        """
        if max_order is None:
            max_order = max(self.clusters[cid]['order'] for cid in self.valid_weights)
        
        results = {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values)
        }
        
        # Store partial sums by order for convergence analysis
        self.partial_sums = {
            'energy': [],
            'specific_heat': [],
            'entropy': []
        }
        
        current_sums = {prop: np.zeros_like(self.temp_values) for prop in results}
        
        for order in range(0, max_order + 1):
            order_contribution = {prop: np.zeros_like(self.temp_values) for prop in results}
            
            for cluster_id in self.valid_weights:
                if self.clusters[cluster_id]['order'] == order:
                    mult = self.clusters[cluster_id]['multiplicity']
                    for prop in results:
                        order_contribution[prop] += mult * self.weights[prop][cluster_id]
            
            for prop in results:
                current_sums[prop] += order_contribution[prop]
                self.partial_sums[prop].append(current_sums[prop].copy())
            
            print(f"Order {order}: Energy contribution norm = {np.linalg.norm(order_contribution['energy']):.6e}")
        
        # Convert to arrays
        for prop in results:
            self.partial_sums[prop] = np.array(self.partial_sums[prop])
            results[prop] = current_sums[prop]
        
        return results
    
    def euler_resummation(self, partial_sums, l=3):
        """Apply Euler transformation for series acceleration."""
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else 0.0
        
        increments = [partial_sums[0]]
        for i in range(1, n):
            increments.append(partial_sums[i] - partial_sums[i-1])
        
        l_use = min(l, max(2, n - 3))
        
        if n <= l_use:
            return partial_sums[-1]
        
        bare_sum = partial_sums[n - l_use - 1] if n - l_use - 1 >= 0 else np.zeros_like(partial_sums[-1])
        
        tail_increments = increments[n - l_use:]
        
        diff_triangle = [tail_increments]
        for k in range(len(tail_increments) - 1):
            prev_row = diff_triangle[-1]
            next_row = [prev_row[i+1] - prev_row[i] for i in range(len(prev_row) - 1)]
            if len(next_row) == 0:
                break
            diff_triangle.append(next_row)
        
        euler_tail = np.zeros_like(partial_sums[-1])
        for k, diff_row in enumerate(diff_triangle):
            if len(diff_row) > 0:
                euler_tail += diff_row[0] / (2**(k+1))
        
        return bare_sum + euler_tail
    
    def wynn_epsilon(self, partial_sums):
        """
        Apply Wynn's epsilon algorithm for series acceleration.
        
        Wynn's epsilon algorithm constructs a table:
        ε_{-1}^{(n)} = 0
        ε_0^{(n)} = S_n  (partial sums)
        ε_{k+1}^{(n)} = ε_{k-1}^{(n+1)} + 1/(ε_k^{(n+1)} - ε_k^{(n)})
        
        The even columns ε_{2k}^{(n)} converge to the limit faster than the
        original series.  Odd columns contain reciprocal differences and are
        NOT sequence estimates.
        
        Singularity handling: when a denominator is near-zero, the entry is
        set to a large sentinel (1e50) so that subsequent iterations also
        detect it.  The final answer is taken from the highest even column
        at row 0 whose value is not blown up.
        
        Supports both scalar sequences and array-valued sequences (one value
        per temperature point).
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else 0.0
        
        SENTINEL = 1e50
        
        # Detect whether entries are scalars or arrays
        first = partial_sums[0]
        is_array = hasattr(first, '__len__')
        
        # Build epsilon table: eps[j][i] where j = column index + 1
        #   j=0 → column k=-1 (zeros)
        #   j=1 → column k=0  (partial sums)
        #   j=2 → column k=1  (reciprocal differences)
        #   ...
        if is_array:
            shape = np.asarray(first).shape
            eps = np.zeros((n + 1, n) + shape)
            for i in range(n):
                eps[1, i] = np.asarray(partial_sums[i])
            
            for j in range(2, n + 1):
                for i in range(n - j + 1):
                    diff = eps[j-1, i+1] - eps[j-1, i]
                    result = np.full(shape, SENTINEL)
                    ok = np.abs(diff) > 1e-15 * (1.0 + np.abs(eps[j-1, i]))
                    if np.any(ok):
                        result[ok] = eps[j-2, i+1][ok] + 1.0 / diff[ok]
                    eps[j, i] = result
            
            # Best estimate: highest even column (j=1,3,5,...) at row 0
            # that is not blown up (element-wise fallback).
            # Start with raw sum S_{n-1} as fallback (better than S_0=0).
            best = np.asarray(partial_sums[-1]).copy()
            for j in range(3, n + 1, 2):
                ok = np.abs(eps[j, 0]) < SENTINEL * 0.1
                best[ok] = eps[j, 0][ok]
            return best
        else:
            # Scalar path
            eps = np.zeros((n + 1, n))
            for i in range(n):
                s = partial_sums[i]
                eps[1, i] = s[0] if hasattr(s, '__len__') else s
            
            for j in range(2, n + 1):
                for i in range(n - j + 1):
                    diff = eps[j-1, i+1] - eps[j-1, i]
                    if abs(diff) < 1e-15 * (1.0 + abs(eps[j-1, i])):
                        eps[j, i] = SENTINEL
                    else:
                        eps[j, i] = eps[j-2, i+1] + 1.0 / diff
            
            # Best estimate: highest even column at row 0 that isn't blown up.
            # Fall back to raw sum S_{n-1} if all Wynn columns blow up.
            s_last = partial_sums[-1]
            best = s_last[0] if hasattr(s_last, '__len__') else s_last
            for j in range(3, n + 1, 2):
                if abs(eps[j, 0]) < SENTINEL * 0.1:
                    best = eps[j, 0]
            return best
    
    def wynn_epsilon_multi_start(self, partial_sums, max_starts=None):
        """Apply Wynn's epsilon from multiple starting indices and take the median.
        
        For a sequence S_0, S_1, ..., S_{n-1}, we apply Wynn starting from
        S_k for k = 0, 1, ..., max_starts-1.  Taking the median across
        starting points is more robust than single-start Wynn when one
        starting point happens to hit a near-singular denominator.
        
        Uses n//2 starting points (leaving at least 3 terms each) to get a
        robust median, instead of n//3 which gives too few samples.
        
        Reference: Khatami & Rigol, Phys. Rev. B 83, 134431 (2011).
        """
        n = len(partial_sums)
        if n < 3:
            return partial_sums[-1] if n > 0 else 0.0
        
        # Use up to n//2 starting points, each needing at least 3 terms
        if max_starts is None:
            max_starts = max(1, n - 2)  # all valid starting points
        max_starts = min(max_starts, max(1, n - 2))
        
        estimates = []
        for k_start in range(max_starts):
            sub_seq = partial_sums[k_start:]
            if len(sub_seq) < 3:
                break
            est = self.wynn_epsilon(sub_seq)
            estimates.append(est)
        
        if not estimates:
            return partial_sums[-1]
        
        # Take element-wise median across starting points
        estimates_arr = np.array(estimates)
        if estimates_arr.ndim == 1:
            return np.median(estimates_arr)
        else:
            return np.median(estimates_arr, axis=0)
    
    def brezinski_theta(self, partial_sums):
        """Brezinski's theta algorithm for series acceleration.
        
        Unlike Wynn epsilon, the theta algorithm uses ALL terms regardless
        of whether n is even or odd.  It constructs a three-column recurrence:
        
            θ_{-1}^{(n)} = 0
            θ_0^{(n)}    = S_n
            ω_{k}^{(n)}  = θ_{k-1}^{(n+1)} + 1/(θ_{k}^{(n+1)} - θ_{k}^{(n)})
            θ_{k+1}^{(n)}= ω_{k}^{(n+1)} + (θ_{k}^{(n+2)} - θ_{k}^{(n+1)}) / 
                           ((θ_{k}^{(n+2)} - θ_{k}^{(n+1)}) / (θ_{k}^{(n+1)} - θ_{k}^{(n)}) - 1)
        
        Equivalently, the theta algorithm consumes 3 terms per "level" and
        each level produces a higher-order estimate.  The best estimate uses
        all n terms.
        
        Ref: Brezinski, J. Comput. Appl. Math. 122, 223 (2000).
        
        Supports both scalar and array-valued sequences.
        """
        n = len(partial_sums)
        if n < 3:
            return partial_sums[-1] if n > 0 else 0.0
        
        SENTINEL = 1e50
        first = partial_sums[0]
        is_array = hasattr(first, '__len__')
        
        if is_array:
            shape = np.asarray(first).shape
            # theta[k][i] and omega[k][i]
            theta = {}
            for i in range(n):
                theta[(-1, i)] = np.zeros(shape)
                theta[(0, i)] = np.asarray(partial_sums[i]).copy()
            
            max_k = (n - 1) // 2  # each theta level consumes 2 terms
            for k in range(max_k):
                # omega_k^(i) = theta_{k-1}^(i+1) + 1/(theta_k^(i+1) - theta_k^(i))
                omega = {}
                for i in range(n - 2*k - 1):
                    diff = theta[(k, i+1)] - theta[(k, i)]
                    result = np.full(shape, SENTINEL)
                    ok = np.abs(diff) > 1e-15 * (1 + np.abs(theta[(k, i)]))
                    if np.any(ok):
                        result[ok] = theta[(k-1, i+1)][ok] + 1.0 / diff[ok]
                    omega[(k, i)] = result
                
                # theta_{k+1}^(i) from omega
                for i in range(n - 2*k - 2):
                    diff_theta = theta[(k, i+2)] - theta[(k, i+1)]
                    diff_theta_prev = theta[(k, i+1)] - theta[(k, i)]
                    result = np.full(shape, SENTINEL)
                    # ratio = diff_theta / diff_theta_prev
                    ok_denom = np.abs(diff_theta_prev) > 1e-15 * (1 + np.abs(theta[(k, i)]))
                    ratio = np.zeros(shape)
                    ratio[ok_denom] = diff_theta[ok_denom] / diff_theta_prev[ok_denom]
                    factor_denom = ratio - 1.0
                    ok = ok_denom & (np.abs(factor_denom) > 1e-15)
                    if np.any(ok):
                        result[ok] = omega[(k, i+1)][ok] + diff_theta[ok] / factor_denom[ok]
                    theta[(k+1, i)] = result
            
            # Best estimate: highest k at row 0 that isn't blown up
            best = np.asarray(partial_sums[-1]).copy()
            for k in range(1, max_k + 1):
                if (k, 0) in theta:
                    ok = np.abs(theta[(k, 0)]) < SENTINEL * 0.1
                    best[ok] = theta[(k, 0)][ok]
            return best
        else:
            # Scalar path
            theta = {}
            for i in range(n):
                theta[(-1, i)] = 0.0
                s = partial_sums[i]
                theta[(0, i)] = s[0] if hasattr(s, '__len__') else s
            
            max_k = (n - 1) // 2
            for k in range(max_k):
                omega = {}
                for i in range(n - 2*k - 1):
                    diff = theta[(k, i+1)] - theta[(k, i)]
                    if abs(diff) < 1e-15 * (1 + abs(theta[(k, i)])):
                        omega[(k, i)] = SENTINEL
                    else:
                        omega[(k, i)] = theta[(k-1, i+1)] + 1.0 / diff
                
                for i in range(n - 2*k - 2):
                    diff_theta = theta[(k, i+2)] - theta[(k, i+1)]
                    diff_theta_prev = theta[(k, i+1)] - theta[(k, i)]
                    if abs(diff_theta_prev) < 1e-15 * (1 + abs(theta[(k, i)])):
                        theta[(k+1, i)] = SENTINEL
                    else:
                        ratio = diff_theta / diff_theta_prev
                        if abs(ratio - 1.0) < 1e-15:
                            theta[(k+1, i)] = SENTINEL
                        else:
                            theta[(k+1, i)] = omega[(k, i+1)] + diff_theta / (ratio - 1.0)
            
            s_last = partial_sums[-1]
            best = s_last[0] if hasattr(s_last, '__len__') else s_last
            for k in range(1, max_k + 1):
                if (k, 0) in theta and abs(theta[(k, 0)]) < SENTINEL * 0.1:
                    best = theta[(k, 0)]
            return best
    
    def iterated_aitken(self, partial_sums):
        """Iterated Aitken Δ² (Shanks) transformation.
        
        Aitken's Δ² method accelerates convergence of a sequence {S_n}:
            S'_n = S_n - (S_{n+1} - S_n)² / (S_{n+2} - 2 S_{n+1} + S_n)
        
        Iterated Aitken applies this repeatedly.  Each application consumes
        2 terms: n terms → n-2 transformed terms.  After ⌊(n-1)/2⌋ levels,
        one estimate remains.  Unlike Wynn, every term contributes at every
        level regardless of parity.
        
        Supports both scalar and array inputs.
        """
        n = len(partial_sums)
        if n < 3:
            return partial_sums[-1] if n > 0 else 0.0
        
        SENTINEL = 1e50
        first = partial_sums[0]
        is_array = hasattr(first, '__len__')
        
        if is_array:
            shape = np.asarray(first).shape
            current = [np.asarray(s).copy() for s in partial_sums]
            
            while len(current) >= 3:
                new_seq = []
                for i in range(len(current) - 2):
                    denom = current[i+2] - 2*current[i+1] + current[i]
                    numer = (current[i+1] - current[i])**2
                    result = current[i].copy()
                    ok = np.abs(denom) > 1e-15 * (1 + np.abs(current[i]))
                    if np.any(ok):
                        result[ok] = current[i][ok] - numer[ok] / denom[ok]
                    # Guard against blowup
                    bad = np.abs(result) > SENTINEL * 0.1
                    result[bad] = current[i][bad]
                    new_seq.append(result)
                current = new_seq
            
            return current[-1]
        else:
            current = []
            for s in partial_sums:
                current.append(s[0] if hasattr(s, '__len__') else s)
            
            while len(current) >= 3:
                new_seq = []
                for i in range(len(current) - 2):
                    denom = current[i+2] - 2*current[i+1] + current[i]
                    numer = (current[i+1] - current[i])**2
                    if abs(denom) < 1e-15 * (1 + abs(current[i])):
                        new_seq.append(current[i])
                    else:
                        val = current[i] - numer / denom
                        if abs(val) > SENTINEL * 0.1:
                            new_seq.append(current[i])
                        else:
                            new_seq.append(val)
                current = new_seq
            
            return current[-1]
    
    def pade_approximant(self, partial_sums):
        """Padé-inspired resummation from NLCE partial sums.
        
        Given partial sums S_0, S_1, ..., S_{n-1}, treat the increments
        a_k = S_k - S_{k-1} as "series coefficients" and form the Padé
        approximant [L/M] of the generating function f(x) = Σ a_k x^k
        evaluated at x=1.
        
        We use a balanced approximant: L = ⌊(n-1)/2⌋, M = n-1-L, so that
        all n coefficients are used regardless of parity.
        
        The Padé approximant is computed via the Wynn rho algorithm on the
        increments, which is equivalent to solving the Padé equations but
        more numerically stable.
        
        Supports both scalar and array inputs.
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else 0.0
        
        SENTINEL = 1e50
        first = partial_sums[0]
        is_array = hasattr(first, '__len__')
        
        if is_array:
            shape = np.asarray(first).shape
            # Increments: a_0 = S_0, a_k = S_k - S_{k-1}
            increments = [np.asarray(partial_sums[0]).copy()]
            for k in range(1, n):
                increments.append(np.asarray(partial_sums[k]) - np.asarray(partial_sums[k-1]))
            
            # Build Padé table via epsilon algorithm on the cumulative sums
            # of increments (which are just the partial sums themselves).
            # Use the Shanks/Padé connection: the [L/M] Padé at x=1 equals
            # the Shanks transform e_M(S_L+M) when L+M = n-1.
            # 
            # For balanced [L/M], we want M = ⌊(n-1)/2⌋.
            # The Shanks e_M transform is exactly Wynn ε_{2M}^{(0)} applied
            # to the first 2M+1 terms... but we want to use ALL n terms.
            #
            # Instead, use the iterated Shanks approach: apply Aitken Δ² 
            # repeatedly to the increments' partial sums.  This is equivalent
            # to the diagonal Padé approximant and uses all terms.
            
            # Actually, the cleanest Padé approach for NLCE:
            # Solve the linear system for [L/M] directly.
            L = (n - 1) // 2
            M = n - 1 - L
            
            # P(x)/Q(x) where P has degree L, Q has degree M, Q(0)=1
            # f(x) Q(x) - P(x) = O(x^n) where f(x) = Σ a_k x^k
            # We need: Σ_{j=0}^{M} q_j a_{k-j} = 0 for k = L+1, ..., L+M
            # with q_0 = 1
            
            # Build per-temperature Padé (vectorized)
            a = np.array([np.asarray(inc) for inc in increments])  # (n, *shape)
            original_shape = shape
            # Flatten spatial dims for batch processing
            a_flat = a.reshape(n, -1)  # (n, n_pts)
            n_pts = a_flat.shape[1]
            
            result = np.asarray(partial_sums[-1]).copy().ravel()
            
            for pt in range(n_pts):
                a_pt = a_flat[:, pt]
                # Build M×M Toeplitz-like system for denominator coefficients
                # Σ_{j=1}^{M} q_j * a_{L+1+i-j} = -a_{L+1+i} for i=0..M-1
                if M == 0:
                    result[pt] = np.sum(a_pt)
                    continue
                mat = np.zeros((M, M))
                rhs = np.zeros(M)
                for i in range(M):
                    for j in range(M):
                        idx = L + 1 + i - (j + 1)
                        if 0 <= idx < n:
                            mat[i, j] = a_pt[idx]
                    rhs[i] = -a_pt[L + 1 + i] if L + 1 + i < n else 0.0
                
                try:
                    q = np.linalg.solve(mat, rhs)
                except np.linalg.LinAlgError:
                    continue  # keep raw sum fallback
                
                # Q(1) = 1 + Σ q_j
                Q1 = 1.0 + np.sum(q)
                if abs(Q1) < 1e-15:
                    continue
                
                # P(x) = Σ_{k=0}^{L} p_k x^k where p_k = a_k + Σ_{j=1}^{min(k,M)} q_j a_{k-j}
                P1 = 0.0
                for k in range(L + 1):
                    p_k = a_pt[k]
                    for j in range(1, min(k, M) + 1):
                        p_k += q[j-1] * a_pt[k - j]
                    P1 += p_k  # x=1
                
                val = P1 / Q1
                if abs(val) < SENTINEL * 0.1:
                    result[pt] = val
            
            return result.reshape(original_shape)
        else:
            # Scalar path
            S = []
            for s in partial_sums:
                S.append(s[0] if hasattr(s, '__len__') else s)
            
            a = [S[0]]
            for k in range(1, n):
                a.append(S[k] - S[k-1])
            
            L = (n - 1) // 2
            M = n - 1 - L
            
            if M == 0:
                return sum(a)
            
            mat = np.zeros((M, M))
            rhs = np.zeros(M)
            for i in range(M):
                for j in range(M):
                    idx = L + 1 + i - (j + 1)
                    if 0 <= idx < n:
                        mat[i, j] = a[idx]
                rhs[i] = -a[L + 1 + i] if L + 1 + i < n else 0.0
            
            try:
                q = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                return S[-1]
            
            Q1 = 1.0 + np.sum(q)
            if abs(Q1) < 1e-15:
                return S[-1]
            
            P1 = 0.0
            for k in range(L + 1):
                p_k = a[k]
                for j in range(1, min(k, M) + 1):
                    p_k += q[j-1] * a[k - j]
                P1 += p_k
            
            val = P1 / Q1
            return val if abs(val) < SENTINEL * 0.1 else S[-1]
    
    def entropy_derived_specific_heat(self, results):
        """Compute specific heat from C(T) = T dS/dT using the NLCE entropy.
        
        The entropy series converges ~1 order faster than the specific heat
        series for frustrated systems.  Computing the derivative numerically
        from the resummed entropy therefore extends the reliable C(T) window
        to lower temperatures.
        
        Uses log-spaced finite differences for accuracy on a log-T grid.
        """
        T = self.temp_values
        S = results['entropy']
        
        # Use log-T derivative: dS/dT = (1/T) dS/d(lnT)
        # This is more accurate on the typical log-spaced temperature grid
        lnT = np.log(T)
        dS_dlnT = np.gradient(S, lnT)
        
        # C = T dS/dT = dS/d(lnT)
        C_derived = dS_dlnT
        
        return C_derived
    
    def perform_resummed_summation(self, max_order=None, method='euler'):
        """Perform NLCE summation with resummation.
        
        Methods:
            'none'/'direct': No resummation, bare partial sums.
            'euler': Euler transformation (Tang-Khatami-Rigol).
            'wynn': Single-start Wynn epsilon.
            'wynn_multi': Multi-start Wynn epsilon with median.
            'brezinski': Brezinski theta algorithm (no even/odd waste).
            'aitken': Iterated Aitken delta-squared (no even/odd waste).
            'pade': Balanced Pade approximant [L/M] (no even/odd waste).
            'entropy_derived': Resum entropy with Euler, then C = T dS/dT.
        """
        raw_results = self.perform_summation(max_order)
        
        # For entropy_derived, resum entropy first, then derive C
        if method == 'entropy_derived':
            results = {}
            # Resum entropy and energy with Euler (best behaved for integrals)
            for prop in ['energy', 'entropy']:
                resummed = np.zeros_like(self.temp_values)
                for i in range(len(self.temp_values)):
                    seq = self.partial_sums[prop][:, i]
                    resummed[i] = self.euler_resummation(seq)
                results[prop] = resummed
            # Derive C(T) = T dS/dT from the resummed entropy
            results['specific_heat'] = self.entropy_derived_specific_heat(results)
            # Also store the direct-resummed C for comparison
            direct_cv = np.zeros_like(self.temp_values)
            for i in range(len(self.temp_values)):
                seq = self.partial_sums['specific_heat'][:, i]
                direct_cv[i] = self.euler_resummation(seq)
            results['specific_heat_direct'] = direct_cv
            print("Specific heat computed via entropy derivative: C(T) = T dS/dT")
            return results
        
        results = {}
        for prop in ['energy', 'specific_heat', 'entropy']:
            if method == 'euler':
                # Apply Euler at each temperature
                resummed = np.zeros_like(self.temp_values)
                for i in range(len(self.temp_values)):
                    seq = self.partial_sums[prop][:, i]
                    resummed[i] = self.euler_resummation(seq)
                results[prop] = resummed
            elif method == 'wynn':
                # Apply Wynn's epsilon algorithm at each temperature
                resummed = np.zeros_like(self.temp_values)
                for i in range(len(self.temp_values)):
                    seq = [self.partial_sums[prop][j, i] for j in range(len(self.partial_sums[prop]))]
                    seq_arr = [np.array([s]) for s in seq]
                    result = self.wynn_epsilon(seq_arr)
                    resummed[i] = result[0] if hasattr(result, '__len__') else result
                results[prop] = resummed
            elif method == 'wynn_multi':
                # Multi-start Wynn with median
                resummed = self.wynn_epsilon_multi_start(self.partial_sums[prop])
                results[prop] = resummed
            elif method == 'brezinski':
                resummed = self.brezinski_theta(self.partial_sums[prop])
                results[prop] = resummed
            elif method == 'aitken':
                resummed = self.iterated_aitken(self.partial_sums[prop])
                results[prop] = resummed
            elif method == 'pade':
                resummed = self.pade_approximant(self.partial_sums[prop])
                results[prop] = resummed
            else:
                results[prop] = raw_results[prop]
        
        return results
    
    def save_results(self, results, output_dir, max_order):
        """Save NLCE results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Temperatures are already in Kelvin
        temp_output = self.temp_values
        temp_unit = 'K'
        
        # Units for thermodynamic quantities
        if self.SI:
            cv_unit = 'J/(mol*K)'
            s_unit = 'J/(mol*K)'
            e_unit = 'J/mol'
        else:
            cv_unit = 'kB'
            s_unit = 'kB'
            e_unit = 'K'
        
        # Save specific heat
        output_file = os.path.join(output_dir, 'nlc_specific_heat.txt')
        data = np.column_stack([temp_output, results['specific_heat']])
        header = f'Temperature({temp_unit})  Specific_Heat({cv_unit})'
        np.savetxt(output_file, data, header=header, comments='# ')
        print(f"Specific heat saved to {output_file}")
        
        # Save entropy-derived specific heat comparison if available
        if 'specific_heat_direct' in results:
            output_file = os.path.join(output_dir, 'nlc_specific_heat_direct.txt')
            data = np.column_stack([temp_output, results['specific_heat_direct']])
            header = f'Temperature({temp_unit})  Specific_Heat_Direct({cv_unit})'
            np.savetxt(output_file, data, header=header, comments='# ')
            print(f"Direct specific heat (for comparison) saved to {output_file}")
        
        # Save energy
        output_file = os.path.join(output_dir, 'nlc_energy.txt')
        data = np.column_stack([temp_output, results['energy']])
        header = f'Temperature({temp_unit})  Energy({e_unit})'
        np.savetxt(output_file, data, header=header, comments='# ')
        print(f"Energy saved to {output_file}")
        
        # Save entropy
        output_file = os.path.join(output_dir, 'nlc_entropy.txt')
        data = np.column_stack([temp_output, results['entropy']])
        header = f'Temperature({temp_unit})  Entropy({s_unit})'
        np.savetxt(output_file, data, header=header, comments='# ')
        print(f"Entropy saved to {output_file}")
        
        # Save order-by-order results for convergence analysis
        for prop in ['specific_heat', 'energy', 'entropy']:
            output_file = os.path.join(output_dir, f'nlc_{prop}_by_order.txt')
            header = f'Temperature({temp_unit})  ' + '  '.join([f'Order_{i}' for i in range(len(self.partial_sums[prop]))])
            data = np.column_stack([temp_output] + [self.partial_sums[prop][i] for i in range(len(self.partial_sums[prop]))])
            np.savetxt(output_file, data, header=header, comments='# ')
    
    def plot_results(self, results, output_dir, max_order):
        """Plot NLCE results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Temperatures are already in Kelvin
        temp_plot = self.temp_values
        temp_label = 'Temperature (K)'
        
        # Units for thermodynamic quantities
        if self.SI:
            cv_label = 'Specific Heat (J/(mol·K))'
            s_label = 'Entropy (J/(mol·K))'
            e_label = 'Energy (J/mol)'
        else:
            cv_label = 'Specific Heat (k_B)'
            s_label = 'Entropy (k_B)'
            e_label = 'Energy (K)'
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Specific heat
        ax = axes[0, 0]
        ax.semilogx(temp_plot, results['specific_heat'], 'b-', lw=2, label='C(T)')
        if 'specific_heat_direct' in results:
            ax.semilogx(temp_plot, results['specific_heat_direct'], 'b--', lw=1.5, alpha=0.6,
                        label='C(T) direct resum')
            ax.legend(loc='best', fontsize=8)
        ax.set_xlabel(temp_label)
        ax.set_ylabel(cv_label)
        ax.set_title(f'Specific Heat (NLCE order {max_order})')
        ax.grid(True, alpha=0.3)
        
        # Energy
        ax = axes[0, 1]
        ax.semilogx(temp_plot, results['energy'], 'r-', lw=2)
        ax.set_xlabel(temp_label)
        ax.set_ylabel(e_label)
        ax.set_title(f'Energy (NLCE order {max_order})')
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[1, 0]
        ax.semilogx(temp_plot, results['entropy'], 'g-', lw=2)
        ax.set_xlabel(temp_label)
        ax.set_ylabel(s_label)
        ax.set_title(f'Entropy (NLCE order {max_order})')
        ax.grid(True, alpha=0.3)
        
        # Convergence
        ax = axes[1, 1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.partial_sums['specific_heat'])))
        for i, ps in enumerate(self.partial_sums['specific_heat']):
            ax.semilogx(temp_plot, ps, color=colors[i], lw=1.5, label=f'Order {i}')
        ax.set_xlabel(temp_label)
        ax.set_ylabel(cv_label)
        ax.set_title('Order-by-order Convergence')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'nlc_results.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {os.path.join(output_dir, 'nlc_results.png')}")


def main():
    parser = argparse.ArgumentParser(description='NLCE summation for triangular lattice')
    parser.add_argument('--cluster_dir', type=str, required=True, 
                       help='Directory containing cluster information files')
    parser.add_argument('--eigenvalue_dir', type=str, required=True,
                       help='Directory containing ED output files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for NLCE results')
    parser.add_argument('--temp_min', type=float, default=0.01,
                       help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=10.0,
                       help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100,
                       help='Number of temperature points')
    parser.add_argument('--temp_points_file', type=str, default=None,
                       help='File containing explicit temperature points (one per line, in Kelvin). '
                            'Overrides --temp_min/--temp_max/--temp_bins.')
    parser.add_argument('--max_order', type=int, default=None,
                       help='Maximum order for summation')
    parser.add_argument('--measure_spin', action='store_true',
                       help='Compute spin expectation values')
    parser.add_argument('--SI_units', action='store_true',
                       help='Convert to SI units: C,S in J/(mol·K), E in J/mol.')
    parser.add_argument('--resummation', type=str, default='none',
                       choices=['none', 'euler', 'wynn', 'wynn_multi',
                                'brezinski', 'aitken', 'pade', 'entropy_derived'],
                       help='Resummation method: none, euler, wynn, wynn_multi, '
                            'brezinski, aitken, pade, or entropy_derived')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NLCE Summation for Triangular Lattice")
    print("="*80)
    
    print(f"\nAll quantities in Kelvin (eigenvalues and temperatures must be in K)")
    
    if args.SI_units:
        print(f"SI units enabled: C, S in J/(mol·K), E in J/mol")
    
    # Load explicit temperature points if provided
    temp_points = None
    if args.temp_points_file:
        temp_points = np.loadtxt(args.temp_points_file)
        print(f"Using {len(temp_points)} explicit temperature points from {args.temp_points_file}")
    
    # Initialize calculator
    if temp_points is not None:
        nlc = NLCExpansionTriangular(
            cluster_dir=args.cluster_dir,
            eigenvalue_dir=args.eigenvalue_dir,
            measure_spin=args.measure_spin,
            SI_units=args.SI_units,
            temp_points=temp_points
        )
    else:
        nlc = NLCExpansionTriangular(
            cluster_dir=args.cluster_dir,
            eigenvalue_dir=args.eigenvalue_dir,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            num_temps=args.temp_bins,
            measure_spin=args.measure_spin,
            SI_units=args.SI_units
        )
    
    # Read data
    print("\nReading cluster information...")
    nlc.read_clusters()
    print(f"Found {len(nlc.clusters)} clusters")
    
    print("\nReading eigenvalues...")
    nlc.read_eigenvalues()
    
    print("\nReading subcluster information...")
    nlc.read_subcluster_info()
    
    # Calculate weights
    print("\nCalculating NLCE weights...")
    nlc.calculate_weights()
    print(f"Valid weights calculated for {len(nlc.valid_weights)} clusters")
    
    # Determine max order
    max_order = args.max_order
    if max_order is None:
        max_order = max(nlc.clusters[cid]['order'] for cid in nlc.valid_weights)
    
    # Perform summation
    print(f"\nPerforming NLCE summation up to order {max_order}...")
    if args.resummation != 'none':
        print(f"Using {args.resummation} resummation")
        results = nlc.perform_resummed_summation(max_order, method=args.resummation)
    else:
        results = nlc.perform_summation(max_order)
    
    # Save results
    print("\nSaving results...")
    nlc.save_results(results, args.output_dir, max_order)
    
    # Plot results
    print("\nPlotting results...")
    nlc.plot_results(results, args.output_dir, max_order)
    
    print("\n" + "="*80)
    print("NLCE summation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
