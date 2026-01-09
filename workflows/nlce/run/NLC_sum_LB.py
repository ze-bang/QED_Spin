#!/usr/bin/env python3
"""
NLC_sum_LB.py - Lanczos-Boosted NLCE Summation

Implements the Lanczos-boosted NLCE method (Bhattaram & Khatami):
- Small clusters: Full ED with all eigenvalues
- Large clusters: Partial Lanczos with N_low lowest eigenvalues

Key difference from standard NLCE:
- Thermodynamic quantities are computed using truncated sums over low-energy eigenstates
- No stochastic noise (unlike FTLM)
- Deterministic and controlled approximation

The truncation error is controlled by ensuring E_{N_low} - E_0 >> kT for target temperatures.

Reference: Bhattaram & Khatami, arXiv:2310.XXXXX
"""

import os
import sys
import numpy as np
import glob
import re
from collections import defaultdict
import argparse
import logging

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("Warning: h5py not installed. HDF5 file reading will not be available.")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class LanczosBoostNLCExpansion:
    """
    Lanczos-Boosted NLCE calculator.
    
    For each cluster, we have either:
    - Full spectrum (small clusters): All eigenvalues
    - Truncated spectrum (large clusters): Only N_low lowest eigenvalues
    
    Thermodynamics are computed as truncated sums:
        Z ≈ Σ_{n=1}^{N_low} exp(-βE_n)
        <A> ≈ Σ_{n=1}^{N_low} <n|A|n> exp(-βE_n) / Z
    
    This is valid when exp(-β(E_{N_low} - E_0)) << 1, i.e., when the
    omitted high-energy states are Boltzmann-suppressed.
    """
    
    def __init__(self, cluster_dir, eigenvalue_dir, temp_min, temp_max, num_temps, 
                 measure_spin=False, SI_units=False, lb_energy_tolerance=10.0):
        """
        Initialize the LB-NLCE calculator.
        
        Args:
            cluster_dir: Directory containing cluster information files
            eigenvalue_dir: Directory containing eigenvalue files from ED calculations
            temp_min, temp_max, num_temps: Temperature grid parameters
            measure_spin: Whether to include spin observables
            SI_units: Whether to output in SI units
            lb_energy_tolerance: Warn if E_max - E_0 < tolerance * T for any T in range
        """
        self.cluster_dir = cluster_dir
        self.eigenvalue_dir = eigenvalue_dir
        self.SI = SI_units
        self.measure_spin = measure_spin
        self.lb_energy_tolerance = lb_energy_tolerance
        
        # Temperature grid (logarithmic for better low-T resolution)
        self.temp_values = np.logspace(np.log10(temp_min), np.log10(temp_max), num_temps)
        
        self.clusters = {}  # {cluster_id: {order, multiplicity, eigenvalues, is_truncated, ...}}
        self.weights = {}   # Calculated weights for each cluster and property
        self.subcluster_info = {}
        
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
                    multiplicity = float(line.split(":")[-1].strip())
                elif line.startswith("# Number of vertices:"):
                    num_vertices = int(line.split(":")[1].strip())
                    
                if multiplicity is not None and num_vertices is not None:
                    break
            
            if multiplicity is None or num_vertices is None:
                logging.warning(f"Missing metadata for cluster {cluster_id}")
                continue
                
            self.clusters[cluster_id] = {
                'order': order,
                'multiplicity': multiplicity,
                'num_vertices': num_vertices,
                'file_path': file_path,
                'eigenvalues': None,
                'hilbert_dim': 2 ** num_vertices,
                'is_truncated': False,  # Will be updated when reading eigenvalues
            }
    
    def read_eigenvalues(self):
        """
        Read eigenvalues for each cluster.
        
        Determines if spectrum is truncated by comparing number of eigenvalues
        to expected Hilbert space dimension.
        """
        for cluster_id in self.clusters:
            cluster_info = self.clusters[cluster_id]
            cluster_output_dir = os.path.join(
                self.eigenvalue_dir, 
                f"cluster_{cluster_id}_order_{cluster_info['order']}/output"
            )
            
            eigenvalues = None
            
            # Try HDF5 file first
            h5_file = os.path.join(cluster_output_dir, "ed_results.h5")
            if HAS_H5PY and os.path.exists(h5_file):
                try:
                    with h5py.File(h5_file, 'r') as f:
                        if '/eigendata/eigenvalues' in f:
                            eigenvalues = f['/eigendata/eigenvalues'][:]
                except Exception as e:
                    logging.warning(f"Error reading HDF5 for cluster {cluster_id}: {e}")
            
            # Fall back to text file
            if eigenvalues is None:
                eigenvalue_file = os.path.join(cluster_output_dir, "eigenvalues.txt")
                if os.path.exists(eigenvalue_file):
                    with open(eigenvalue_file, 'r') as f:
                        eigenvalues = np.array([float(line.strip()) for line in f if line.strip()])
            
            if eigenvalues is None:
                logging.warning(f"No eigenvalue data for cluster {cluster_id}")
                continue
            
            # Sort eigenvalues (should already be sorted, but ensure)
            eigenvalues = np.sort(eigenvalues)
            
            self.clusters[cluster_id]['eigenvalues'] = eigenvalues
            
            # Check if spectrum is truncated
            hilbert_dim = cluster_info['hilbert_dim']
            n_eigenvalues = len(eigenvalues)
            
            # Allow some tolerance for symmetry-reduced Hilbert space
            self.clusters[cluster_id]['is_truncated'] = (n_eigenvalues < hilbert_dim * 0.99)
            
            if self.clusters[cluster_id]['is_truncated']:
                energy_window = eigenvalues[-1] - eigenvalues[0]
                logging.info(f"Cluster {cluster_id}: Truncated spectrum ({n_eigenvalues}/{hilbert_dim} states, "
                           f"energy window = {energy_window:.4f})")
    
    def check_truncation_validity(self):
        """
        Check if truncated spectra are valid for the target temperature range.
        
        For each truncated cluster, warn if E_max - E_0 < lb_energy_tolerance * T_max.
        """
        logging.info("\n" + "="*60)
        logging.info("Checking truncation validity for LB-NLCE")
        logging.info("="*60)
        
        T_min = self.temp_values[0]
        T_max = self.temp_values[-1]
        
        warnings = []
        
        for cluster_id, info in self.clusters.items():
            if not info['is_truncated'] or info['eigenvalues'] is None:
                continue
            
            eigenvalues = info['eigenvalues']
            energy_window = eigenvalues[-1] - eigenvalues[0]
            
            # For accurate thermodynamics, we need E_max - E_0 >> kT
            # Rule of thumb: E_max - E_0 > 10*T ensures truncation error < e^-10 ≈ 5e-5
            required_window = self.lb_energy_tolerance * T_max
            
            if energy_window < required_window:
                msg = (f"Cluster {cluster_id} (order {info['order']}): "
                      f"Energy window {energy_window:.3f} < {required_window:.3f} (= {self.lb_energy_tolerance}*T_max). "
                      f"May need more eigenvalues for T > {energy_window/self.lb_energy_tolerance:.3f}")
                warnings.append(msg)
                logging.warning(msg)
        
        if not warnings:
            logging.info(f"All truncated spectra satisfy E_max - E_0 > {self.lb_energy_tolerance}*T_max = "
                        f"{self.lb_energy_tolerance * T_max:.3f}")
        
        return warnings
    
    def calculate_thermodynamic_quantities_truncated(self, eigenvalues, is_truncated=False):
        """
        Calculate thermodynamic quantities from (possibly truncated) eigenvalues.
        
        For LB-NLCE, the partition function is approximated as:
            Z ≈ Σ_{n=1}^{N_low} exp(-βE_n)
        
        This is valid when exp(-β(E_{N_low} - E_0)) << 1.
        
        Returns:
            Dictionary with 'energy', 'specific_heat', 'entropy', and truncation info
        """
        n_states = len(eigenvalues)
        ground_state_energy = eigenvalues[0]
        
        # Shift eigenvalues for numerical stability
        shifted_eigenvalues = eigenvalues - ground_state_energy
        
        results = {
            'energy': np.zeros_like(self.temp_values),
            'specific_heat': np.zeros_like(self.temp_values),
            'entropy': np.zeros_like(self.temp_values),
            'truncation_error_estimate': np.zeros_like(self.temp_values),
        }
        
        for i, temp in enumerate(self.temp_values):
            beta = 1.0 / temp
            
            # Compute Boltzmann weights
            exp_terms = np.exp(-beta * shifted_eigenvalues)
            Z_truncated = np.sum(exp_terms)
            
            # Energy (using original eigenvalues)
            energy = np.sum(eigenvalues * exp_terms) / Z_truncated
            
            # Energy squared for specific heat
            energy_squared = np.sum(eigenvalues**2 * exp_terms) / Z_truncated
            
            # Specific heat C = β² * (<E²> - <E>²)
            specific_heat = beta**2 * (energy_squared - energy**2)
            
            # Entropy S = ln(Z) + β<E> (shifted version: ln(Z_shifted) + β<E-E_0>)
            entropy = np.log(Z_truncated) + beta * (energy - ground_state_energy)
            
            # Estimate truncation error: weight of omitted states
            # Upper bound: assume all omitted states have energy ≥ E_max
            if is_truncated:
                truncation_error = np.exp(-beta * shifted_eigenvalues[-1])
            else:
                truncation_error = 0.0
            
            if self.SI:
                kB_NA = 6.02214076e23 * 1.380649e-23  # J/K per mole
                specific_heat *= kB_NA
                entropy *= kB_NA
                energy *= kB_NA
            
            results['energy'][i] = energy
            results['specific_heat'][i] = specific_heat
            results['entropy'][i] = entropy
            results['truncation_error_estimate'][i] = truncation_error
        
        return results
    
    def read_subcluster_info(self):
        """Read subcluster information from subclusters_info.txt."""
        subcluster_file = os.path.join(self.cluster_dir, 'subclusters_info.txt')
        
        if not os.path.exists(subcluster_file):
            logging.warning(f"Subcluster info file not found: {subcluster_file}")
            return
        
        with open(subcluster_file, 'r') as f:
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
                        
                elif 'Subclusters:' in line and current_cluster is not None:
                    subclusters_str = line.split('Subclusters:')[-1].strip()
                    pairs = re.findall(r'\((\d+),\s*(\d+)\)', subclusters_str)
                    for subcluster_id, multiplicity in pairs:
                        self.subcluster_info[current_cluster]['subclusters'][int(subcluster_id)] = int(multiplicity)
    
    def get_subclusters(self, cluster_id):
        """Get subclusters with multiplicities for a given cluster."""
        if cluster_id in self.subcluster_info:
            return self.subcluster_info[cluster_id]['subclusters']
        
        # Fallback to order-based heuristic
        subclusters = {}
        order = self.clusters[cluster_id]['order']
        for cid, data in self.clusters.items():
            if data['order'] < order:
                subclusters[cid] = 1
        return subclusters
    
    def _topological_sort_clusters(self):
        """Sort clusters in dependency order for weight calculation."""
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
            logging.warning(f"Dependency cycle detected! Remaining: {remaining}")
            for cid in sorted(remaining, key=lambda c: (self.clusters[c]['order'], c)):
                sorted_clusters.append((cid, self.clusters[cid]['order']))
        
        return sorted_clusters
    
    def calculate_weights(self):
        """Calculate NLCE weights for all clusters using truncated thermodynamics."""
        if not self.subcluster_info:
            self.read_subcluster_info()
        
        sorted_clusters = self._topological_sort_clusters()
        
        logging.info(f"\nCalculating weights for {len(sorted_clusters)} clusters")
        
        self.weights = {
            'energy': {},
            'specific_heat': {},
            'entropy': {},
        }
        
        self.valid_weights = set()
        self.truncation_info = {}  # Track truncation status for each cluster
        
        for cluster_id, order in sorted_clusters:
            if self.clusters[cluster_id]['eigenvalues'] is None:
                logging.debug(f"Cluster {cluster_id}: SKIPPED (no eigenvalues)")
                continue
            
            subclusters = self.get_subclusters(cluster_id)
            
            # Check if all subclusters have valid weights
            missing = [s for s in subclusters.keys() if s not in self.valid_weights]
            if missing:
                logging.debug(f"Cluster {cluster_id}: SKIPPED (missing subclusters {missing})")
                continue
            
            # Calculate thermodynamic quantities using truncated spectrum
            is_truncated = self.clusters[cluster_id]['is_truncated']
            quantities = self.calculate_thermodynamic_quantities_truncated(
                self.clusters[cluster_id]['eigenvalues'],
                is_truncated=is_truncated
            )
            
            self.truncation_info[cluster_id] = {
                'is_truncated': is_truncated,
                'n_eigenvalues': len(self.clusters[cluster_id]['eigenvalues']),
                'max_truncation_error': np.max(quantities['truncation_error_estimate']),
            }
            
            # Calculate weights: W(c) = P(c) - Σ Y_cs × W(s)
            for prop in ['energy', 'specific_heat', 'entropy']:
                weight = quantities[prop].copy()
                for subcluster_id, multiplicity in subclusters.items():
                    if subcluster_id in self.weights[prop]:
                        weight -= self.weights[prop][subcluster_id] * multiplicity
                self.weights[prop][cluster_id] = weight
            
            self.valid_weights.add(cluster_id)
    
    def sum_nlc(self, order_cutoff=None, resummation_method='auto'):
        """
        Perform NLCE summation with the computed weights.
        
        Returns:
            Dictionary with summed thermodynamic properties
        """
        properties = ['energy', 'specific_heat', 'entropy']
        results = {prop: np.zeros_like(self.temp_values) for prop in properties}
        results['partial_sums'] = {prop: {} for prop in properties}
        
        for prop in properties:
            if prop not in self.weights:
                continue
            
            # Sum by order
            sum_by_order = defaultdict(lambda: np.zeros_like(self.temp_values))
            
            for cluster_id, weight in self.weights[prop].items():
                order = self.clusters[cluster_id]['order']
                if order_cutoff is not None and order > order_cutoff:
                    continue
                sum_by_order[order] += weight * self.clusters[cluster_id]['multiplicity']
            
            if sum_by_order:
                max_order = max(sum_by_order.keys())
                partial_sums = []
                cumulative = np.zeros_like(self.temp_values)
                
                for order in range(max_order + 1):
                    if order in sum_by_order:
                        cumulative = cumulative + sum_by_order[order]
                    partial_sums.append(cumulative.copy())
                    results['partial_sums'][prop][order] = cumulative.copy()
                
                # Apply resummation if needed
                partial_sums = np.array(partial_sums)
                results[prop] = self._apply_resummation(partial_sums, resummation_method, prop)
        
        results['temperatures'] = self.temp_values
        return results
    
    def _apply_resummation(self, partial_sums, method, prop_name):
        """Apply resummation method to partial sums."""
        if len(partial_sums) == 0:
            return np.zeros_like(self.temp_values)
        
        if method == 'auto' or method == 'direct':
            return partial_sums[-1]
        elif method == 'euler':
            return self._euler_resummation(partial_sums)
        else:
            logging.info(f"{prop_name}: Using direct summation")
            return partial_sums[-1]
    
    def _euler_resummation(self, partial_sums, l=3):
        """Apply Euler transformation for series acceleration."""
        n = len(partial_sums)
        if n < 2:
            return partial_sums[-1] if n > 0 else np.zeros_like(self.temp_values)
        
        increments = [partial_sums[0]]
        for i in range(1, n):
            increments.append(partial_sums[i] - partial_sums[i-1])
        
        l_use = min(l, max(2, n - 3))
        
        if n <= l_use:
            return partial_sums[-1]
        
        bare_sum = partial_sums[n - l_use - 1] if n - l_use - 1 >= 0 else np.zeros_like(self.temp_values)
        
        tail_increments = increments[n - l_use:]
        
        diff_triangle = [tail_increments]
        for k in range(len(tail_increments) - 1):
            prev_row = diff_triangle[-1]
            next_row = [prev_row[i+1] - prev_row[i] for i in range(len(prev_row) - 1)]
            if len(next_row) == 0:
                break
            diff_triangle.append(next_row)
        
        euler_tail = np.zeros_like(self.temp_values)
        for k, diff_row in enumerate(diff_triangle):
            if len(diff_row) > 0:
                euler_tail += diff_row[0] / (2**(k+1))
        
        return bare_sum + euler_tail
    
    def report_truncation_summary(self):
        """Print summary of truncation status for all clusters."""
        logging.info("\n" + "="*60)
        logging.info("LB-NLCE Truncation Summary")
        logging.info("="*60)
        
        n_truncated = sum(1 for info in self.truncation_info.values() if info['is_truncated'])
        n_full = len(self.truncation_info) - n_truncated
        
        logging.info(f"Full spectrum clusters: {n_full}")
        logging.info(f"Truncated spectrum clusters: {n_truncated}")
        
        if n_truncated > 0:
            max_errors = [info['max_truncation_error'] for info in self.truncation_info.values() 
                         if info['is_truncated']]
            logging.info(f"Max truncation error estimate: {max(max_errors):.2e}")
            logging.info(f"Mean truncation error estimate: {np.mean(max_errors):.2e}")
    
    def save_results(self, output_dir, results):
        """Save NLCE results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        output_file = os.path.join(output_dir, 'lb_nlce_results.dat')
        header = "# Lanczos-Boosted NLCE Results\n"
        header += "# T\tenergy\tspecific_heat\tentropy\n"
        
        data = np.column_stack([
            results['temperatures'],
            results['energy'],
            results['specific_heat'],
            results['entropy']
        ])
        
        np.savetxt(output_file, data, header=header, fmt='%.10e')
        logging.info(f"Saved results to {output_file}")
        
        # Save partial sums for convergence analysis
        for prop in ['energy', 'specific_heat', 'entropy']:
            partial_file = os.path.join(output_dir, f'lb_nlce_partial_sums_{prop}.dat')
            if results['partial_sums'][prop]:
                orders = sorted(results['partial_sums'][prop].keys())
                header = f"# Partial sums for {prop} by order\n# T\t" + "\t".join([f"order_{o}" for o in orders])
                
                data = [results['temperatures']]
                for order in orders:
                    data.append(results['partial_sums'][prop][order])
                
                np.savetxt(partial_file, np.column_stack(data), header=header, fmt='%.10e')
        
        # Save truncation info
        trunc_file = os.path.join(output_dir, 'lb_nlce_truncation_info.dat')
        with open(trunc_file, 'w') as f:
            f.write("# Cluster truncation information\n")
            f.write("# cluster_id\tn_eigenvalues\tis_truncated\tmax_truncation_error\n")
            for cid, info in sorted(self.truncation_info.items()):
                f.write(f"{cid}\t{info['n_eigenvalues']}\t{info['is_truncated']}\t{info['max_truncation_error']:.6e}\n")
    
    def plot_results(self, output_dir, results):
        """Generate plots of NLCE results."""
        if not HAS_MATPLOTLIB:
            logging.warning("Matplotlib not available, skipping plots")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        T = results['temperatures']
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Lanczos-Boosted NLCE Results")
        
        # Energy
        axs[0, 0].plot(T, results['energy'], 'b-', linewidth=2)
        axs[0, 0].set_xlabel('Temperature')
        axs[0, 0].set_ylabel('Energy per site')
        axs[0, 0].set_xscale('log')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].set_title('Energy')
        
        # Specific heat
        axs[0, 1].plot(T, results['specific_heat'], 'r-', linewidth=2)
        axs[0, 1].set_xlabel('Temperature')
        axs[0, 1].set_ylabel('Specific Heat')
        axs[0, 1].set_xscale('log')
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].set_title('Specific Heat')
        
        # Entropy
        axs[1, 0].plot(T, results['entropy'], 'g-', linewidth=2)
        axs[1, 0].set_xlabel('Temperature')
        axs[1, 0].set_ylabel('Entropy per site')
        axs[1, 0].set_xscale('log')
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].set_title('Entropy')
        
        # Convergence: plot partial sums for specific heat
        if results['partial_sums']['specific_heat']:
            orders = sorted(results['partial_sums']['specific_heat'].keys())
            colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
            for i, order in enumerate(orders):
                axs[1, 1].plot(T, results['partial_sums']['specific_heat'][order], 
                              color=colors[i], label=f'Order {order}', alpha=0.7)
            axs[1, 1].set_xlabel('Temperature')
            axs[1, 1].set_ylabel('Specific Heat')
            axs[1, 1].set_xscale('log')
            axs[1, 1].grid(True, alpha=0.3)
            axs[1, 1].legend(loc='upper right', fontsize=8)
            axs[1, 1].set_title('Convergence by Order')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lb_nlce_results.png'), dpi=150)
        plt.close()
        
        logging.info(f"Saved plot to {os.path.join(output_dir, 'lb_nlce_results.png')}")


def main():
    parser = argparse.ArgumentParser(
        description='Lanczos-Boosted NLCE Summation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Lanczos-Boosted NLCE uses partial Lanczos diagonalization for large clusters,
computing only the lowest N_low eigenvalues. This is deterministic (no FTLM noise)
and ideal for low-to-intermediate temperatures.

Reference: Bhattaram & Khatami, arXiv:2310.XXXXX
        """
    )
    
    parser.add_argument('--cluster_dir', type=str, required=True,
                       help='Directory containing cluster info files')
    parser.add_argument('--eigenvalue_dir', type=str, required=True,
                       help='Directory containing ED output')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    
    parser.add_argument('--temp_min', type=float, default=0.01,
                       help='Minimum temperature')
    parser.add_argument('--temp_max', type=float, default=10.0,
                       help='Maximum temperature')
    parser.add_argument('--temp_bins', type=int, default=100,
                       help='Number of temperature points')
    
    parser.add_argument('--order_cutoff', type=int, default=None,
                       help='Maximum NLCE order to include')
    parser.add_argument('--resummation_method', type=str, default='auto',
                       choices=['auto', 'direct', 'euler'],
                       help='Resummation method for series acceleration')
    
    parser.add_argument('--lb_energy_tolerance', type=float, default=10.0,
                       help='Warn if energy window < tolerance * T_max')
    
    parser.add_argument('--SI_units', action='store_true',
                       help='Output in SI units')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("="*60)
    logging.info("Lanczos-Boosted NLCE Summation")
    logging.info("="*60)
    
    # Initialize calculator
    nlce = LanczosBoostNLCExpansion(
        cluster_dir=args.cluster_dir,
        eigenvalue_dir=args.eigenvalue_dir,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        num_temps=args.temp_bins,
        SI_units=args.SI_units,
        lb_energy_tolerance=args.lb_energy_tolerance,
    )
    
    # Run workflow
    logging.info("Reading cluster information...")
    nlce.read_clusters()
    logging.info(f"Found {len(nlce.clusters)} clusters")
    
    logging.info("Reading eigenvalues...")
    nlce.read_eigenvalues()
    
    # Check truncation validity
    nlce.check_truncation_validity()
    
    logging.info("Reading subcluster info...")
    nlce.read_subcluster_info()
    
    logging.info("Calculating weights...")
    nlce.calculate_weights()
    
    # Report truncation summary
    nlce.report_truncation_summary()
    
    logging.info("Performing NLCE summation...")
    results = nlce.sum_nlc(
        order_cutoff=args.order_cutoff,
        resummation_method=args.resummation_method
    )
    
    # Save results
    logging.info("Saving results...")
    nlce.save_results(args.output_dir, results)
    
    if args.plot:
        logging.info("Generating plots...")
        nlce.plot_results(args.output_dir, results)
    
    logging.info("="*60)
    logging.info("LB-NLCE completed successfully!")
    logging.info("="*60)


if __name__ == "__main__":
    main()
