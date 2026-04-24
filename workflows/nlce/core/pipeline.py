"""Pipeline abstraction + registry.

A *Pipeline* describes one ED strategy (full diagonalization,
Lanczos-boosted eigensolve, FTLM, mTPQ, ...). It owns:

  * which ``./ED`` method to invoke per cluster (and any per-cluster
    auto-promotion rules: e.g. FTLM uses full ED for small clusters);
  * which extra CLI flags it adds to the unified NLCE CLI
    (``--ftlm_samples``, ``--krylov_dim``, ``--lb_n_eigenvalues``, ...);
  * how to translate the user's CLI args + one cluster's site count
    into an :class:`EDOptions`;
  * which NLCE-summation kernel to invoke after ED finishes.

Concrete subclasses live in ``workflows.nlce.pipelines`` and register
themselves with :func:`register_pipeline`.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from .ed_runner import EDOptions

__all__ = [
    "Pipeline",
    "register_pipeline",
    "get_pipeline",
    "list_pipelines",
]


_REGISTRY: dict[str, type["Pipeline"]] = {}


class Pipeline(ABC):
    """Abstract base class for one ED strategy."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    # -- CLI hook ---------------------------------------------------------

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Inject pipeline-specific CLI flags. Default: no-op.

        Subclasses override to add ``--ftlm_samples``, ``--krylov_dim``,
        ``--lb_n_eigenvalues``, etc.
        """

    # -- Per-cluster ED config -------------------------------------------

    @abstractmethod
    def make_ed_options(
        self,
        args: argparse.Namespace,
        num_sites: int,
    ) -> EDOptions:
        """Build the :class:`EDOptions` for one cluster.

        Pipelines may inspect ``num_sites`` to make adaptive decisions
        (e.g. FTLM uses full ED for tiny clusters; Lanczos-boost picks
        the Krylov dim from cluster size).
        """

    # -- Summation hook --------------------------------------------------

    @abstractmethod
    def summation_command(
        self,
        args: argparse.Namespace,
        cluster_info_dir: str,
        ed_dir: str,
        nlc_dir: str,
        order_cutoff: int,
    ) -> Optional[list[str]]:
        """Return the argv for the post-ED NLCE summation step.

        Returning ``None`` skips the summation step (some pipelines
        defer summation to a separate driver).
        """

    # -- Optional hooks --------------------------------------------------

    def needs_thermo(self, args: argparse.Namespace) -> bool:
        """Whether the pipeline always wants ``--thermo`` enabled."""
        return False

    def needs_basis_precompute(self, args: argparse.Namespace) -> bool:
        """Whether the pipeline requires the per-cluster orbit basis
        precomputation step (only relevant when ``--streaming-symmetry``
        is on). Default: True iff the user passed ``--streaming-symmetry``."""
        return bool(getattr(args, "streaming_symmetry", False))

    def extra_env(
        self,
        args: argparse.Namespace,
        num_sites: int,
    ) -> Optional[dict[str, str]]:
        """Pipeline-specific environment overrides for one ED invocation.

        Default: None. Triangular geometries override this to set
        ``OMP_NUM_THREADS=1`` for tiny clusters that thrash on
        many-core machines.
        """
        return None


def register_pipeline(pipeline_cls: type[Pipeline]) -> type[Pipeline]:
    """Register a Pipeline subclass under its ``name``."""
    if not pipeline_cls.name:
        raise ValueError(f"Pipeline {pipeline_cls.__name__} must set `name`")
    if pipeline_cls.name in _REGISTRY and _REGISTRY[pipeline_cls.name] is not pipeline_cls:
        raise ValueError(
            f"Pipeline name {pipeline_cls.name!r} already registered to "
            f"{_REGISTRY[pipeline_cls.name].__name__}"
        )
    _REGISTRY[pipeline_cls.name] = pipeline_cls
    return pipeline_cls


def get_pipeline(name: str) -> Pipeline:
    """Look up a registered pipeline by name and instantiate it."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown pipeline {name!r}. Available: {sorted(_REGISTRY)}. "
            f"Did you forget to ``import workflows.nlce.pipelines``?"
        )
    return _REGISTRY[name]()


def list_pipelines() -> list[str]:
    """Return the sorted names of all registered pipelines."""
    return sorted(_REGISTRY)
