"""Backward-compatibility re-exports.

This module used to host the shared NLCE infrastructure (logging,
cluster discovery, ED-CLI builder, subprocess driver, HDF5/text
fallback readers). Those have been moved into
:mod:`workflows.nlce.core` as part of the package modernization
(see :mod:`workflows.nlce`'s docstring for the new layout).

The original public symbols are re-exported from here so that
external scripts that still write::

    from workflows.nlce._common import (
        EDOptions, build_ed_command, run_ed_subprocess, ...
    )

continue to work unchanged. New code should import directly from
:mod:`workflows.nlce.core` (or use the unified
:mod:`workflows.nlce.cli`) rather than going through this shim.
"""

from .core.ed_runner import (
    DEFAULT_ED_PATH,
    EDOptions,
    build_ed_command,
    run_ed_subprocess,
)
from .core.io import (
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

__all__ = [
    "HAS_H5PY",
    "DEFAULT_ED_PATH",
    "ClusterEntry",
    "EDOptions",
    "check_gpu_available",
    "setup_logging",
    "get_cluster_files",
    "get_num_sites",
    "count_sites_in_info_file",
    "build_ed_command",
    "run_ed_subprocess",
    "load_thermo_dataset",
    "load_tpq_thermo_dataset",
]
