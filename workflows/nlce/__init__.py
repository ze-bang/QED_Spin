"""NLCE (Numerical Linked Cluster Expansion) -- a standalone modern
NLCE package built on top of the C++ ``./ED`` toolkit.

Public package layout
---------------------

* :mod:`workflows.nlce.core`
    Abstractions: :class:`Geometry`, :class:`Pipeline`, registries,
    :class:`NLCEWorkflow` orchestrator, and the canonical bridge to the
    ``./ED`` binary (:class:`EDOptions`, :func:`build_ed_command`,
    :func:`run_ed_subprocess`, plus I/O helpers).

* :mod:`workflows.nlce.geometries`
    Concrete lattice implementations: ``pyrochlore``,
    ``triangular_site``, ``triangular_triangle``. Importing this
    subpackage triggers all geometry registrations.

* :mod:`workflows.nlce.pipelines`
    Concrete ED-strategy implementations: ``full_ed``,
    ``lanczos_boost``, ``ftlm``. Importing this subpackage triggers
    all pipeline registrations.

* :mod:`workflows.nlce.cli`
    Unified CLI: ``python -m workflows.nlce``. Selects one geometry
    and one pipeline at the command line, composes them via
    :class:`NLCEWorkflow`, runs the four-step pipeline.

* :mod:`workflows.nlce.prep`
    Cluster generators (run by ``Geometry`` implementations).

* :mod:`workflows.nlce.run`
    NLCE summation kernels (run by ``Pipeline`` implementations) plus
    the legacy per-lattice driver scripts (now thin shims over the
    unified CLI).

Quick API entry::

    from workflows.nlce.core import (
        get_geometry, get_pipeline, NLCEWorkflow,
    )
    from workflows.nlce import geometries, pipelines  # trigger registration
    geom = get_geometry("pyrochlore")
    pipe = get_pipeline("ftlm")
    # ... build args via argparse, then NLCEWorkflow(geom, pipe, args).run()

Backward compatibility
----------------------

The old :mod:`workflows.nlce._common` module is preserved as a
re-export shim of the new modules in :mod:`workflows.nlce.core`, so
external scripts importing ``from workflows.nlce._common import ...``
continue to work.
"""

# core: abstractions + workflow orchestrator + ED bridge
from . import core  # noqa: F401

# legacy shim — re-exports from .core (kept for downstream callers)
from . import _common  # noqa: F401

__all__ = ["core", "_common"]


def _autodiscover_extensions() -> None:
    """Trigger registration of all bundled geometries and pipelines.

    Called on import so that downstream consumers can do ``from
    workflows.nlce.core import get_geometry`` without first having
    to import the geometry/pipeline subpackages explicitly.
    """
    from . import geometries  # noqa: F401  -- registration side-effect
    from . import pipelines  # noqa: F401  -- registration side-effect


_autodiscover_extensions()
