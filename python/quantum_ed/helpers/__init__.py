"""quantum_ed.helpers: re-exports of the legacy ``edlib`` geometry utilities.

These modules were originally developed under the ``edlib`` namespace.
Importing ``quantum_ed.helpers`` exposes them under the new
``quantum_ed.*`` namespace while keeping ``edlib`` as a compatibility
alias (see ``edlib.__init__``). New code should import from
``quantum_ed.helpers``.

This module is intentionally lazy: heavy imports (e.g. lattice helpers
that pull SymPy or sage-like dependencies) only happen when accessed.
"""

from importlib import import_module
from typing import Any

_ALIAS_MAP = {
    "hdf5_io":                          "edlib.hdf5_io",
    "automorphism_finder":              "edlib.automorphism_finder",
    "helper_cluster":                   "edlib.helper_cluster",
    "helper_cluster_triangular":        "edlib.helper_cluster_triangular",
    "helper_cluster_triangular_triangle_based":
                                        "edlib.helper_cluster_triangular_triangle_based",
    "helper_honeycomb":                 "edlib.helper_honeycomb",
    "helper_honeycomb_BCAO":            "edlib.helper_honeycomb_BCAO",
    "helper_honeycomb_c3":              "edlib.helper_honeycomb_c3",
    "helper_honeycomb_c3_BCAO":         "edlib.helper_honeycomb_c3_BCAO",
    "helper_kagome_bfg":                "edlib.helper_kagome_bfg",
    "helper_kagome_bfg_sqrt3":          "edlib.helper_kagome_bfg_sqrt3",
    "helper_pyrochlore":                "edlib.helper_pyrochlore",
    "helper_pyrochlore_super":          "edlib.helper_pyrochlore_super",
}

__all__ = list(_ALIAS_MAP.keys())


def __getattr__(name: str) -> Any:
    target = _ALIAS_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'quantum_ed.helpers' has no attribute '{name}'")
    module = import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(__all__)
