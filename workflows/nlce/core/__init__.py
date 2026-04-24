"""Core NLCE abstractions: geometries, pipelines, and the workflow orchestrator.

This subpackage defines the *only* extension points that downstream
research code is expected to subclass against:

* :class:`Geometry` -- describes a lattice (which generator to invoke,
  which Hamiltonian helper to invoke, which model parameters are
  meaningful). One concrete subclass per supported lattice lives in
  ``workflows.nlce.geometries``.

* :class:`Pipeline` -- describes an ED strategy (which `./ED` method
  to use, which extra CLI flags to thread through, which NLCE
  summation kernel to call afterwards). One concrete subclass per
  ED strategy lives in ``workflows.nlce.pipelines``.

* :class:`NLCEWorkflow` -- composes one ``Geometry`` × one ``Pipeline``
  and runs the canonical four-step pipeline (clusters → Hamiltonians
  → ED → summation).

* :class:`EDOptions` / :func:`build_ed_command` / :func:`run_ed_subprocess`
  -- the only legal way for a ``Pipeline`` to talk to the C++ ``./ED``
  binary. Centralizing this means CLI flag changes only need to be
  audited in one place.

* I/O helpers (:func:`get_cluster_files`, :func:`load_thermo_dataset`,
  ...) -- the canonical readers for the on-disk schema.
"""

from .ed_runner import (
    DEFAULT_ED_PATH,
    EDOptions,
    build_ed_command,
    run_ed_subprocess,
)
from .io import (
    HAS_H5PY,
    ClusterEntry,
    check_gpu_available,
    count_sites_in_info_file,
    get_cluster_files,
    get_num_sites,
    load_thermo_dataset,
    load_tpq_thermo_dataset,
    setup_logging,
)
from .geometry import Geometry, register_geometry, get_geometry, list_geometries
from .pipeline import Pipeline, register_pipeline, get_pipeline, list_pipelines
from .workflow import NLCEWorkflow

__all__ = [
    # ed_runner
    "DEFAULT_ED_PATH",
    "EDOptions",
    "build_ed_command",
    "run_ed_subprocess",
    # io
    "HAS_H5PY",
    "ClusterEntry",
    "check_gpu_available",
    "count_sites_in_info_file",
    "get_cluster_files",
    "get_num_sites",
    "load_thermo_dataset",
    "load_tpq_thermo_dataset",
    "setup_logging",
    # geometry / pipeline / workflow
    "Geometry",
    "register_geometry",
    "get_geometry",
    "list_geometries",
    "Pipeline",
    "register_pipeline",
    "get_pipeline",
    "list_pipelines",
    "NLCEWorkflow",
]
