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
    
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min, temp_max, num_temps, 
                 measure_spin=False, SI_units=False):
        """
        Initialize the NLC expansion calculator for triangular lattice.
        
        All quantities are in Kelvin:
            - Hamiltonian couplings (and hence eigenvalues) must be in Kelvin
            - Temperatures are in Kelvin
            - Energy output is in Kelvin (or J/mol if SI)
        
        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_min: Minimum temperature in Kelvin
            temp_max: Maximum temperature in Kelvin
            num_temps: Number of temperature points
            measure_spin: Whether to compute spin expectation values
            SI_units: Convert to SI units (J/(mol·K) for C and S, J/mol for E)
                      
        SI Unit Conversion:
            - Specific heat: C_SI = R × C where R = 8.314 J/(mol·K)
            - Entropy: S_SI = R × S  
            - Energy: E_SI = R × E [J/mol] (E in Kelvin)
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir
        
        self.SI = SI_units
        self.measure_spin = measure_spin
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
        """Perform NLCE summation up to specified order."""
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
        
        for order in range(1, max_order + 1):
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
        
        The even columns ε_{2k}^{(n)} converge to the limit faster than the original series.
        The best estimate is typically ε_{2*floor((n-1)/2)}^{(0)} for n partial sums.
        
        This is particularly effective for alternating series like NLCE.
        """
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else 0.0
        
        # Initialize epsilon table
        # We need columns -1, 0, 1, 2, ..., n-1
        # epsilon[k][i] = ε_k^{(i)}
        # Column -1 is all zeros
        # Column 0 is partial sums
        
        # For numerical stability, we work with arrays
        epsilon = {}
        
        # Column -1: all zeros
        for i in range(n):
            epsilon[(-1, i)] = np.zeros_like(partial_sums[0])
        
        # Column 0: partial sums
        for i in range(n):
            epsilon[(0, i)] = partial_sums[i].copy()
        
        # Build subsequent columns
        # ε_{k+1}^{(i)} = ε_{k-1}^{(i+1)} + 1/(ε_k^{(i+1)} - ε_k^{(i)})
        for k in range(0, n - 1):
            for i in range(n - k - 1):
                diff = epsilon[(k, i + 1)] - epsilon[(k, i)]
                
                # Handle numerical instability when diff is very small
                # Use element-wise operations for arrays
                if np.isscalar(diff):
                    if np.abs(diff) < 1e-15:
                        epsilon[(k + 1, i)] = epsilon[(k - 1, i + 1)]
                    else:
                        epsilon[(k + 1, i)] = epsilon[(k - 1, i + 1)] + 1.0 / diff
                else:
                    # Array case
                    result = epsilon[(k - 1, i + 1)].copy()
                    mask = np.abs(diff) > 1e-15
                    if np.any(mask):
                        result[mask] = result[mask] + 1.0 / diff[mask]
                    epsilon[(k + 1, i)] = result
        
        # The best approximation is from even columns
        # For n partial sums, we can compute up to column n-1
        # The even columns 0, 2, 4, ... give approximations
        # Return the highest even column at row 0
        best_col = 2 * ((n - 1) // 2)
        if best_col >= 0 and (best_col, 0) in epsilon:
            return epsilon[(best_col, 0)]
        else:
            return partial_sums[-1]
    
    def perform_resummed_summation(self, max_order=None, method='euler'):
        """Perform NLCE summation with resummation."""
        raw_results = self.perform_summation(max_order)
        
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
                    # Convert to array of scalars for wynn_epsilon
                    seq_arr = [np.array([s]) for s in seq]
                    result = self.wynn_epsilon(seq_arr)
                    resummed[i] = result[0] if hasattr(result, '__len__') else result
                results[prop] = resummed
            else:
                results[prop] = raw_results[prop]
        
        return results
    
    def save_results(self, results, output_dir, max_order):
        """Save NLCE results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
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
            header = f'Temperature({temp_unit})  ' + '  '.join([f'Order_{i+1}' for i in range(len(self.partial_sums[prop]))])
            data = np.column_stack([temp_output] + [self.partial_sums[prop][i] for i in range(len(self.partial_sums[prop]))])
            np.savetxt(output_file, data, header=header, comments='# ')
    
    def plot_results(self, results, output_dir, max_order):
        """Plot NLCE results."""
        os.makedirs(output_dir, exist_ok=True)
        
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
        ax.semilogx(temp_plot, results['specific_heat'], 'b-', lw=2)
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
            ax.semilogx(temp_plot, ps, color=colors[i], lw=1.5, label=f'Order {i+1}')
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
    parser.add_argument('--max_order', type=int, default=None,
                       help='Maximum order for summation')
    parser.add_argument('--measure_spin', action='store_true',
                       help='Compute spin expectation values')
    parser.add_argument('--SI_units', action='store_true',
                       help='Convert to SI units: C,S in J/(mol·K), E in J/mol.')
    parser.add_argument('--resummation', type=str, default='none',
                       choices=['none', 'euler', 'wynn'],
                       help='Resummation method (none, euler, or wynn)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("NLCE Summation for Triangular Lattice")
    print("="*80)
    
    print(f"\nAll quantities in Kelvin (eigenvalues and temperatures must be in K)")
    if args.SI_units:
        print(f"SI units enabled: C, S in J/(mol·K), E in J/mol")
    
    # Initialize calculator
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
    if args.resummation == 'euler':
        print("Using Euler resummation")
        results = nlc.perform_resummed_summation(max_order, method='euler')
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
