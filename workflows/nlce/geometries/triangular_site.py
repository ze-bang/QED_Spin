"""Triangular-lattice NLCE geometry, *site-based* expansion.

Order = number of *sites*. This is the conservative expansion (more
clusters per order, slower convergence) but works for any model
formulated on the triangular lattice.

Wraps:

* ``workflows/nlce/prep/generate_triangular_clusters.py``
* ``python/edlib/helper_cluster_triangular.py``

Models supported:

* ``xxz_j1j2``    -- J1-J2 XXZ
* ``kitaev``      -- J-K-Γ-Γ' Kitaev-Heisenberg
* ``anisotropic`` -- YbMgGaO4-type anisotropic exchange
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

from ..core import Geometry, register_geometry
from . import _paths


def _add_triangular_arguments(parser: argparse.ArgumentParser) -> None:
    """Shared arg group for the two triangular geometries."""
    g = parser.add_argument_group("triangular model parameters")
    g.add_argument("--J1", type=float, default=1.0,
                   help="Nearest-neighbour exchange (or J for kitaev model)")
    g.add_argument("--J2", type=float, default=0.0,
                   help="Next-nearest-neighbour exchange (or K for kitaev model)")
    g.add_argument("--Jz_ratio", type=float, default=1.0,
                   help="Jz/Jxy ratio for XXZ model")
    g.add_argument("--h", type=float, default=0.0, help="Magnetic field strength")
    g.add_argument("--field_dir", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                   help="Field direction; default is out-of-plane (z)")
    g.add_argument("--model", type=str, default="xxz_j1j2",
                   choices=["xxz_j1j2", "kitaev", "anisotropic"],
                   help="Spin model")
    # Anisotropic model
    g.add_argument("--Jzz", type=float, default=None, help="J_zz (anisotropic)")
    g.add_argument("--Jpm", type=float, default=None, help="J_± (anisotropic)")
    g.add_argument("--Jpmpm", type=float, default=None, help="J_±± (anisotropic)")
    g.add_argument("--Jzpm", type=float, default=None, help="J_z± (anisotropic)")
    # Kitaev
    g.add_argument("--Gamma", type=float, default=None, help="Γ (kitaev)")
    g.add_argument("--Gamma_prime", type=float, default=None, help="Γ' (kitaev)")
    # g-tensor
    g.add_argument("--g_ab", type=float, default=2.0, help="In-plane g-factor")
    g.add_argument("--g_c", type=float, default=2.0, help="Out-of-plane g-factor")
    # Cluster-generator visualization
    g.add_argument("--visualize", action="store_true",
                   help="Emit visualizations during cluster generation.")
    # Triangular-specific NLCE knobs
    g.add_argument("--symm_threshold", type=int, default=13,
                   help="Site threshold for using --symm at the ED step (default 13)")
    g.add_argument("--streaming-symmetry", dest="streaming_symmetry",
                   action="store_true",
                   help="Use --streaming-symmetry kernel with cached orbit basis.")
    g.add_argument("--skip_basis_precompute", action="store_true",
                   help="Skip the orbit-basis precompute step (assumes cache exists). "
                        "Only meaningful with --streaming-symmetry.")


def _run_triangular_helper(args: argparse.Namespace, cluster_file_path: str,
                           ham_subdir: str, cluster_id: int) -> bool:
    cmd = [
        sys.executable,
        _paths.TRIANGULAR_HELPER,
        "--J1", str(args.J1),
        "--J2", str(args.J2),
        "--h", str(args.h),
        "--field_dir", str(args.field_dir[0]), str(args.field_dir[1]), str(args.field_dir[2]),
        "--output_dir", ham_subdir,
        "--cluster_file", cluster_file_path,
        "--model", args.model,
        "--Jz_ratio", str(args.Jz_ratio),
    ]
    if args.Jzz is not None:
        cmd += ["--Jzz", str(args.Jzz)]
    if args.Jpm is not None:
        cmd += ["--Jpm", str(args.Jpm)]
    if args.Jpmpm is not None:
        cmd += ["--Jpmpm", str(args.Jpmpm)]
    if args.Jzpm is not None:
        cmd += ["--Jzpm", str(args.Jzpm)]
    if args.Gamma is not None:
        cmd += ["--Gamma", str(args.Gamma)]
    if args.Gamma_prime is not None:
        cmd += ["--Gamma_prime", str(args.Gamma_prime)]
    cmd += ["--g_ab", str(args.g_ab), "--g_c", str(args.g_c)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(
            "helper_cluster_triangular.py failed for cluster %d: %s",
            cluster_id, e,
        )
        if e.stderr:
            logging.error("stderr: %s", e.stderr.decode("utf-8", errors="replace"))
        return False


def _precompute_basis_for_cluster(
    args: argparse.Namespace, cluster_id: int, order: int, ham_subdir: str,
) -> bool:
    """Run ``./ED --precompute-basis`` once for one cluster."""
    import glob
    if not os.path.exists(ham_subdir):
        logging.warning("Ham dir missing for cluster %d: %s", cluster_id, ham_subdir)
        return False

    basis_cache = os.path.join(ham_subdir, "basis_cache")
    if os.path.isdir(basis_cache) and glob.glob(os.path.join(basis_cache, "*.h5")):
        return True  # already cached

    site_info = glob.glob(os.path.join(ham_subdir, "*_site_info.dat"))
    if not site_info:
        logging.warning("No site info file in %s", ham_subdir)
        return False
    num_sites = 0
    with open(site_info[0]) as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                num_sites += 1

    cmd = [
        args.ed_executable,
        ham_subdir,
        "--precompute-basis",
        f"--num_sites={num_sites}",
        "--spin_length=0.5",
    ]
    env = os.environ.copy()
    env["ED_PYTHON"] = sys.executable
    if num_sites <= 8:
        env["OMP_NUM_THREADS"] = "1"
    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Basis precompute failed for cluster %d: %s", cluster_id, e)
        if e.stderr:
            logging.error("stderr: %s", e.stderr.decode("utf-8", errors="replace"))
        return False


@register_geometry
class TriangularSite(Geometry):
    name = "triangular_site"
    description = "Triangular lattice; site-based NLCE expansion (order = sites)."

    default_temp_min = 0.1
    default_temp_max = 10.0
    default_temp_bins = 100
    default_min_order = 1
    default_max_order = 8

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        _add_triangular_arguments(parser)

    def generate_clusters(
        self, args: argparse.Namespace, order: int, cluster_dir: str
    ) -> bool:
        cmd = [
            sys.executable,
            _paths.TRIANGULAR_SITE_GENERATOR,
            f"--max_order={order}",
            f"--output_dir={cluster_dir}",
        ]
        if getattr(args, "visualize", False):
            cmd.append("--visualize")
        logging.info("Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logging.error("Triangular (site) cluster generation failed: %s", e)
            return False

    def prepare_hamiltonian(
        self, args, cluster_id, order, cluster_file_path, ham_subdir,
    ) -> bool:
        return _run_triangular_helper(args, cluster_file_path, ham_subdir, cluster_id)

    def precompute_basis(self, args, cluster_id, order, ham_subdir) -> bool:
        return _precompute_basis_for_cluster(args, cluster_id, order, ham_subdir)
