"""
HDF5 I/O utilities for exact diagonalization results

This module provides functions to read HDF5 files created by the C++ ED code.
"""

import h5py
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class EDResultsReader:
    """
    Reader for HDF5 files containing exact diagonalization results.
    
    File structure matches the C++ HDF5IO class:
    - /eigendata: Eigenvalues and eigenvectors
    - /thermodynamics: Temperature-dependent observables
    - /correlations: Correlation functions
    - /dynamical: Dynamical response and structure factors
    - /ftlm: FTLM sample data
    - /tpq: TPQ state data
    """
    
    def __init__(self, filepath: str):
        """
        Initialize reader with HDF5 file path.
        
        Args:
            filepath: Path to the HDF5 file
        """
        self.filepath = filepath
        self._file = None
    
    def __enter__(self):
        """Context manager entry."""
        self._file = h5py.File(self.filepath, 'r')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._file is not None:
            self._file.close()
    
    # ============================================================================
    # Eigenvalue/Eigenvector methods
    # ============================================================================
    
    def get_eigenvalues(self) -> np.ndarray:
        """
        Read eigenvalues from HDF5 file.
        
        Returns:
            1D array of eigenvalues
        """
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                return f['/eigendata/eigenvalues'][:]
        return self._file['/eigendata/eigenvalues'][:]
    
    def get_eigenvector(self, index: int) -> np.ndarray:
        """
        Read a single eigenvector from HDF5 file.
        
        Args:
            index: Index of the eigenvector
        
        Returns:
            Complex 1D array representing the eigenvector
        """
        dataset_name = f'/eigendata/eigenvector_{index}'
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                if dataset_name not in f:
                    raise KeyError(f"Eigenvector {index} not found in file")
                data = f[dataset_name][:]
        else:
            if dataset_name not in self._file:
                raise KeyError(f"Eigenvector {index} not found in file")
            data = self._file[dataset_name][:]
        
        # Convert compound dtype to complex array
        return data['real'] + 1j * data['imag']
    
    def get_all_eigenvectors(self) -> List[np.ndarray]:
        """
        Read all eigenvectors from HDF5 file.
        
        Returns:
            List of complex 1D arrays
        """
        eigenvectors = []
        idx = 0
        while True:
            try:
                eigenvectors.append(self.get_eigenvector(idx))
                idx += 1
            except KeyError:
                break
        return eigenvectors
    
    # ============================================================================
    # Thermodynamics methods
    # ============================================================================
    
    def get_temperatures(self) -> np.ndarray:
        """Get temperature array."""
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                return f['/thermodynamics/temperatures'][:]
        return self._file['/thermodynamics/temperatures'][:]
    
    def get_thermodynamic_observable(self, observable_name: str) -> np.ndarray:
        """
        Get thermodynamic observable values.
        
        Args:
            observable_name: Name of observable ('energy', 'entropy', 'specific_heat', etc.)
        
        Returns:
            1D array of observable values vs temperature
        """
        dataset_name = f'/thermodynamics/{observable_name}'
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                if dataset_name not in f:
                    raise KeyError(f"Observable {observable_name} not found")
                return f[dataset_name][:]
        
        if dataset_name not in self._file:
            raise KeyError(f"Observable {observable_name} not found")
        return self._file[dataset_name][:]
    
    def get_all_thermodynamics(self) -> Dict[str, np.ndarray]:
        """
        Get all thermodynamic observables.
        
        Returns:
            Dictionary mapping observable names to arrays
        """
        thermo = {}
        thermo['temperature'] = self.get_temperatures()
        
        # Common observables
        for obs in ['energy', 'entropy', 'specific_heat', 'free_energy', 
                    'magnetization', 'susceptibility']:
            try:
                thermo[obs] = self.get_thermodynamic_observable(obs)
            except KeyError:
                pass  # Observable not present
        
        return thermo
    
    # ============================================================================
    # Correlation functions methods
    # ============================================================================
    
    def get_correlation_matrix(self, correlation_name: str) -> np.ndarray:
        """
        Get correlation matrix (complex 2D array).
        
        Args:
            correlation_name: Name of correlation ('spin_spin', 'density_density', etc.)
        
        Returns:
            Complex 2D array
        """
        real_name = f'/correlations/{correlation_name}_real'
        imag_name = f'/correlations/{correlation_name}_imag'
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                real_part = f[real_name][:]
                imag_part = f[imag_name][:]
        else:
            real_part = self._file[real_name][:]
            imag_part = self._file[imag_name][:]
        
        return real_part + 1j * imag_part
    
    def get_correlation_data(self, dataset_name: str) -> np.ndarray:
        """
        Get 1D correlation data (complex array).
        
        Args:
            dataset_name: Name of dataset
        
        Returns:
            Complex 1D array
        """
        real_name = f'/correlations/{dataset_name}_real'
        imag_name = f'/correlations/{dataset_name}_imag'
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                real_part = f[real_name][:]
                imag_part = f[imag_name][:]
        else:
            real_part = self._file[real_name][:]
            imag_part = self._file[imag_name][:]
        
        return real_part + 1j * imag_part
    
    # ============================================================================
    # Dynamical response methods
    # ============================================================================
    
    def get_frequencies(self) -> np.ndarray:
        """Get frequency array for dynamical response."""
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                return f['/dynamical/frequencies'][:]
        return self._file['/dynamical/frequencies'][:]
    
    def get_dynamical_response(self, operator_name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get dynamical response S(ω) for a given operator.
        
        Args:
            operator_name: Name of operator
        
        Returns:
            Tuple of (frequencies, spectral_function, metadata)
        """
        frequencies = self.get_frequencies()
        dataset_name = f'/dynamical/{operator_name}'
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                spectral = f[dataset_name][:]
                metadata = dict(f[dataset_name].attrs)
        else:
            spectral = self._file[dataset_name][:]
            metadata = dict(self._file[dataset_name].attrs)
        
        return frequencies, spectral, metadata
    
    # ============================================================================
    # FTLM methods
    # ============================================================================
    
    def get_ftlm_sample(self, sample_index: int) -> Dict[str, np.ndarray]:
        """
        Get FTLM sample data.
        
        Args:
            sample_index: Sample index
        
        Returns:
            Dictionary with eigenvalues and observables
        """
        sample_group = f'/ftlm/samples/sample_{sample_index}'
        
        result = {}
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                if sample_group not in f:
                    raise KeyError(f"FTLM sample {sample_index} not found")
                for key in f[sample_group].keys():
                    result[key] = f[sample_group][key][:]
        else:
            if sample_group not in self._file:
                raise KeyError(f"FTLM sample {sample_index} not found")
            for key in self._file[sample_group].keys():
                result[key] = self._file[sample_group][key][:]
        
        return result
    
    def get_all_ftlm_samples(self) -> List[Dict[str, np.ndarray]]:
        """Get all FTLM samples."""
        samples = []
        idx = 0
        while True:
            try:
                samples.append(self.get_ftlm_sample(idx))
                idx += 1
            except KeyError:
                break
        return samples
    
    # ============================================================================
    # TPQ methods
    # ============================================================================
    
    def get_tpq_state(self, sample_index: int, beta: float) -> Tuple[np.ndarray, Dict]:
        """
        Get TPQ state vector.
        
        Args:
            sample_index: Sample index
            beta: Inverse temperature
        
        Returns:
            Tuple of (state_vector, metadata)
        """
        dataset_name = f'/tpq/states/state_{sample_index}_beta_{beta:.6f}'
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                if dataset_name not in f:
                    raise KeyError(f"TPQ state not found: {dataset_name}")
                data = f[dataset_name][:]
                metadata = dict(f[dataset_name].attrs)
        else:
            if dataset_name not in self._file:
                raise KeyError(f"TPQ state not found: {dataset_name}")
            data = self._file[dataset_name][:]
            metadata = dict(self._file[dataset_name].attrs)
        
        # Convert compound dtype to complex array
        state = data['real'] + 1j * data['imag']
        return state, metadata
    
    # ============================================================================
    # Utility methods
    # ============================================================================
    
    def list_datasets(self, group: str = '/') -> List[str]:
        """
        List all datasets in a group.
        
        Args:
            group: Group path (default: root)
        
        Returns:
            List of dataset names
        """
        datasets = []
        
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append(name)
        
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                f[group].visititems(visitor)
        else:
            self._file[group].visititems(visitor)
        
        return datasets
    
    def print_structure(self, group: str = '/', indent: int = 0):
        """
        Print the structure of the HDF5 file.
        
        Args:
            group: Group to start from (default: root)
            indent: Indentation level
        """
        if self._file is None:
            with h5py.File(self.filepath, 'r') as f:
                self._print_group(f[group], indent)
        else:
            self._print_group(self._file[group], indent)
    
    def _print_group(self, group: h5py.Group, indent: int):
        """Helper to recursively print group structure."""
        for key in group.keys():
            item = group[key]
            print('  ' * indent + f'{key}')
            if isinstance(item, h5py.Group):
                self._print_group(item, indent + 1)
            elif isinstance(item, h5py.Dataset):
                print('  ' * (indent + 1) + f'shape: {item.shape}, dtype: {item.dtype}')


# Convenience functions for quick access
def load_eigenvalues(filepath: str) -> np.ndarray:
    """Quickly load eigenvalues from HDF5 file."""
    with EDResultsReader(filepath) as reader:
        return reader.get_eigenvalues()


def load_thermodynamics(filepath: str) -> Dict[str, np.ndarray]:
    """Quickly load all thermodynamic data from HDF5 file."""
    with EDResultsReader(filepath) as reader:
        return reader.get_all_thermodynamics()


def load_eigenvector(filepath: str, index: int) -> np.ndarray:
    """Quickly load a single eigenvector from HDF5 file."""
    with EDResultsReader(filepath) as reader:
        return reader.get_eigenvector(index)


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hdf5_io.py <path_to_hdf5_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    print(f"Reading HDF5 file: {filepath}")
    print("=" * 60)
    
    with EDResultsReader(filepath) as reader:
        print("\nFile structure:")
        print("-" * 60)
        reader.print_structure()
        
        print("\n\nAvailable datasets:")
        print("-" * 60)
        datasets = reader.list_datasets()
        for ds in datasets:
            print(f"  {ds}")
        
        # Try to read eigenvalues
        try:
            eigenvalues = reader.get_eigenvalues()
            print(f"\n\nEigenvalues: {len(eigenvalues)} values")
            print(f"Ground state energy: {eigenvalues[0]}")
            if len(eigenvalues) > 1:
                print(f"First gap: {eigenvalues[1] - eigenvalues[0]}")
        except Exception as e:
            print(f"\nCould not read eigenvalues: {e}")
        
        # Try to read thermodynamics
        try:
            thermo = reader.get_all_thermodynamics()
            print(f"\n\nThermodynamic observables: {list(thermo.keys())}")
            if 'temperature' in thermo:
                print(f"Temperature range: {thermo['temperature'].min():.3f} - {thermo['temperature'].max():.3f}")
        except Exception as e:
            print(f"\nCould not read thermodynamics: {e}")
