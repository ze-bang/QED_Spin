"""Geometry abstraction + registry.

A *Geometry* describes one lattice (pyrochlore, triangular,
honeycomb, kagome, ...). It owns:

  * cluster generation: which standalone script (in ``prep/``) to
    invoke and which CLI args it expects;
  * Hamiltonian preparation: which ``edlib.helper_*`` script to invoke
    per cluster, and which model parameters (``J1/J2``, ``Jxx/Jyy/Jzz``,
    ``Jzz/Jpm/Jpmpm/Jzpm``, ...) it understands;
  * cluster filename conventions: where the cluster ``.dat`` files
    end up in the output tree (the same NLCEWorkflow paths apply
    to every geometry, but each generator writes its own subtree);
  * sensible default temperature ranges.

Concrete subclasses live in ``workflows.nlce.geometries`` and register
themselves with :func:`register_geometry`. Importing
``workflows.nlce.geometries`` triggers all registrations.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

__all__ = [
    "Geometry",
    "register_geometry",
    "get_geometry",
    "list_geometries",
]


_REGISTRY: dict[str, type["Geometry"]] = {}


class Geometry(ABC):
    """Abstract base class for one supported lattice geometry.

    Subclasses must set ``name`` (the registry key, also the CLI value
    for ``--geometry``) and implement the three abstract methods. The
    ``add_arguments`` hook lets each geometry inject its own
    model-parameter CLI flags into the unified NLCE CLI.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    # Default temperature grid (only used when the user doesn't override
    # via --temp_min / --temp_max / --temp_bins).
    default_temp_min: ClassVar[float] = 0.001
    default_temp_max: ClassVar[float] = 20.0
    default_temp_bins: ClassVar[int] = 100

    # Default order span. Each geometry's expansion order means a
    # different thing (sites for triangular_site, triangles for
    # triangular_triangle, tetrahedra for pyrochlore), so the defaults
    # diverge.
    default_min_order: ClassVar[int] = 1
    default_max_order: ClassVar[int] = 4

    # Subdirectory naming used by the generators. Override if a
    # specific generator deviates from the default convention.
    cluster_dir_prefix: ClassVar[str] = "clusters_order_"
    cluster_info_subdir: ClassVar[str] = "cluster_info_order_{order}"
    hamiltonian_dir_prefix: ClassVar[str] = "hamiltonians_order_"
    ed_dir_prefix: ClassVar[str] = "ed_results_order_"
    nlc_dir_prefix: ClassVar[str] = "nlc_results_order_"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Subclasses don't auto-register; concrete geometry modules
        # call ``register_geometry(MyGeom)`` explicitly. This keeps
        # the registration entirely under ``geometries/__init__.py``'s
        # control (lazy, debuggable, easy to override in tests).

    # -- argument injection -------------------------------------------------

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Inject geometry-specific CLI flags into the unified NLCE parser.

        These typically include the model-parameter flags (``--J1``,
        ``--Jxx``, ``--field_dir``, ...) and any geometry-specific
        knobs (``--Jz_ratio``, ``--g_ab``, ``--g_c``, ...).
        """

    # -- cluster generation -------------------------------------------------

    @abstractmethod
    def generate_clusters(
        self,
        args: argparse.Namespace,
        order: int,
        cluster_dir: str,
    ) -> bool:
        """Generate cluster ``.dat`` files for one expansion order.

        Returns True on success. Implementations typically shell out to
        ``prep/generate_*_clusters.py`` via :mod:`subprocess`.
        """

    # -- hamiltonian preparation -------------------------------------------

    @abstractmethod
    def prepare_hamiltonian(
        self,
        args: argparse.Namespace,
        cluster_id: int,
        order: int,
        cluster_file_path: str,
        ham_subdir: str,
    ) -> bool:
        """Produce the per-cluster ``*_site_info.dat`` / ``*_inter_all.dat``
        files inside ``ham_subdir`` for one cluster.

        Returns True on success.
        """

    # -- optional hooks ----------------------------------------------------

    def precompute_basis(
        self,
        args: argparse.Namespace,
        cluster_id: int,
        order: int,
        ham_subdir: str,
    ) -> bool:
        """Optional pre-ED step (e.g., orbit basis precomputation for
        ``--streaming-symmetry``). Default: no-op."""
        return True

    def cluster_info_path(self, base_dir: str, max_order: int) -> str:
        """Return the directory holding the cluster info files for the
        ``--max_order`` run."""
        import os
        return os.path.join(
            base_dir,
            f"{self.cluster_dir_prefix}{max_order}",
            self.cluster_info_subdir.format(order=max_order),
        )


def register_geometry(geometry_cls: type[Geometry]) -> type[Geometry]:
    """Register a Geometry subclass under its ``name``.

    Intended for use as a decorator::

        @register_geometry
        class Pyrochlore(Geometry):
            name = "pyrochlore"
            ...
    """
    if not geometry_cls.name:
        raise ValueError(f"Geometry {geometry_cls.__name__} must set `name`")
    if geometry_cls.name in _REGISTRY and _REGISTRY[geometry_cls.name] is not geometry_cls:
        raise ValueError(
            f"Geometry name {geometry_cls.name!r} already registered to "
            f"{_REGISTRY[geometry_cls.name].__name__}"
        )
    _REGISTRY[geometry_cls.name] = geometry_cls
    return geometry_cls


def get_geometry(name: str) -> Geometry:
    """Look up a registered geometry by name and instantiate it."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown geometry {name!r}. Available: {sorted(_REGISTRY)}. "
            f"Did you forget to ``import workflows.nlce.geometries``?"
        )
    return _REGISTRY[name]()


def list_geometries() -> list[str]:
    """Return the sorted names of all registered geometries."""
    return sorted(_REGISTRY)
