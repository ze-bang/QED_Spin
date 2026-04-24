"""Triangular-lattice NLCE geometry, *triangle-based* expansion.

Order = number of *triangles* (rather than sites). Yields far fewer
clusters per order (OEIS A007854: 1, 1, 3, 5, 12, 35, 98, 299, ...),
which is essential for frustrated-magnet NLCE where convergence is
slow.

Wraps:

* ``workflows/nlce/prep/generate_triangle_nlce_clusters.py``
* ``python/edlib/helper_cluster_triangular.py`` (same helper as the
  site-based variant -- the cluster file format is identical, only
  the generator differs).
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from ..core import Geometry, register_geometry
from . import _paths
from .triangular_site import (
    _add_triangular_arguments,
    _run_triangular_helper,
    _precompute_basis_for_cluster,
)


@register_geometry
class TriangularTriangle(Geometry):
    name = "triangular_triangle"
    description = (
        "Triangular lattice; triangle-based NLCE expansion (order = triangles)."
    )

    default_temp_min = 0.1
    default_temp_max = 10.0
    default_temp_bins = 100
    default_min_order = 1
    default_max_order = 6

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_triangular_arguments(parser)

    def generate_clusters(
        self, args: argparse.Namespace, order: int, cluster_dir: str
    ) -> bool:
        cmd = [
            sys.executable,
            _paths.TRIANGULAR_TRIANGLE_GENERATOR,
            f"--max_order={order}",
            f"--output_dir={cluster_dir}",
        ]
        if getattr(args, "visualize", False):
            cmd.append("--visualize")
        else:
            cmd.append("--no_visualize")
        logging.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error("Triangle-based cluster generation failed: %s", e)
            return False

    def prepare_hamiltonian(
        self, args, cluster_id, order, cluster_file_path, ham_subdir,
    ) -> bool:
        return _run_triangular_helper(args, cluster_file_path, ham_subdir, cluster_id)

    def precompute_basis(self, args, cluster_id, order, ham_subdir) -> bool:
        return _precompute_basis_for_cluster(args, cluster_id, order, ham_subdir)
