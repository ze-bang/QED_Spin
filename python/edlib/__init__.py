# edlib - Exact Diagonalization Library Python Utilities
"""edlib: legacy entry point for the lattice helper modules.

This package is preserved so existing notebooks (`from edlib.helper_pyrochlore
import ...`) keep working. New code should prefer the modern
``quantum_ed.helpers`` aliases (see ``quantum_ed.helpers``) and the
``quantum_ed`` solver bindings.

The C++ pybind11 bindings are exposed under ``quantum_ed`` only -- there is
intentionally no ``edlib`` shim for them, because the legacy package never
exposed a C++ surface.

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
