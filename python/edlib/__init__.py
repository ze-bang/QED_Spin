# edlib - Exact Diagonalization Library Python Utilities
"""
Python utilities for exact diagonalization calculations.

Modules:
    hdf5_io: HDF5 file input/output for eigenvectors, thermodynamics, etc.
    automorphism_finder: Find graph automorphisms for symmetry operations
    helper_cluster: Generic cluster geometry helpers
    helper_honeycomb: Honeycomb lattice helpers
    helper_honeycomb_BCAO: Honeycomb lattice for BCAO materials
    helper_honeycomb_c3: C3-symmetric honeycomb helpers
    helper_honeycomb_c3_BCAO: C3-symmetric honeycomb for BCAO
    helper_kagome_bfg: Kagome lattice helpers
    helper_kagome_bfg_hexcentric: Hexagon-centric Kagome lattice helpers
    helper_non_kramers: Non-Kramers doublet helpers
    helper_pyrochlore: Pyrochlore lattice helpers
    helper_pyrochlore_super: Supercell pyrochlore helpers
"""

__version__ = "0.1.0"

from .hdf5_io import *
from .automorphism_finder import *
