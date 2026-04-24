"""Pyrochlore-lattice NLCE geometry.

Wraps two existing scripts:

* ``workflows/nlce/prep/generate_pyrochlore_clusters.py`` for cluster
  generation (orbits of tetrahedra),
* ``python/edlib/helper_cluster.py`` for per-cluster Hamiltonian
  preparation (XYZ + Zeeman + optional random transverse field).

Order = number of *tetrahedra* (the natural pyrochlore expansion unit).
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys

from ..core import Geometry, register_geometry
from . import _paths


@register_geometry
class Pyrochlore(Geometry):
    name = "pyrochlore"
    description = "Pyrochlore lattice; XYZ + Zeeman + optional random transverse field."

    default_temp_min = 0.001
    default_temp_max = 20.0
    default_temp_bins = 100
    default_min_order = 1
    default_max_order = 4

    cluster_dir_prefix = "clusters_order_"
    cluster_info_subdir = "cluster_info_order_{order}"

    # ------------------------------------------------------------ args

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("pyrochlore model parameters")
        g.add_argument("--Jxx", type=float, default=1.0, help="Jxx coupling")
        g.add_argument("--Jyy", type=float, default=1.0, help="Jyy coupling")
        g.add_argument("--Jzz", type=float, default=1.0, help="Jzz coupling")
        g.add_argument("--h", type=float, default=0.0, help="Magnetic field strength")
        g.add_argument(
            "--field_dir", type=float, nargs=3,
            default=[1 / math.sqrt(3)] * 3,
            help="Field direction (x, y, z); default is the [111] body diagonal",
        )
        g.add_argument(
            "--random_field_width", type=float, default=0.0,
            help="Width of the random transverse field (default 0)",
        )

    # ------------------------------------------------------ cluster gen

    def generate_clusters(
        self, args: argparse.Namespace, order: int, cluster_dir: str
    ) -> bool:
        cmd = [
            sys.executable,
            _paths.PYROCHLORE_GENERATOR,
            f"--max_order={order}",
            f"--output_dir={cluster_dir}",
            "--visualize",
        ]
        logging.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error("Pyrochlore cluster generation failed: %s", e)
            return False

    # ------------------------------------------------------- ham prep

    def prepare_hamiltonian(
        self,
        args: argparse.Namespace,
        cluster_id: int,
        order: int,
        cluster_file_path: str,
        ham_subdir: str,
    ) -> bool:
        cmd = [
            sys.executable,
            _paths.PYROCHLORE_HELPER,
            str(args.Jxx),
            str(args.Jyy),
            str(args.Jzz),
            str(args.h),
            str(args.field_dir[0]),
            str(args.field_dir[1]),
            str(args.field_dir[2]),
            ham_subdir,
            cluster_file_path,
            str(args.random_field_width),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(
                "helper_cluster.py failed for cluster %d: %s",
                cluster_id, e,
            )
            if e.stdout:
                logging.error("stdout: %s", e.stdout.decode("utf-8", errors="replace"))
            if e.stderr:
                logging.error("stderr: %s", e.stderr.decode("utf-8", errors="replace"))
            return False
