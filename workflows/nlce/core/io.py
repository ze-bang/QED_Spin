"""I/O helpers shared by every NLCE driver.

This module owns the conventions for reading what the project writes
to disk (cluster info files, ED output HDF5 / text fallback, TPQ
samples). All path-walking, log setup, and HDF5 fallback logic is
funnelled through here so individual ``Pipeline`` / ``Geometry``
implementations don't reinvent the wheel.

Layout convention (used by every driver):

    <base_dir>/
    ├── clusters_order_<N>/                # generated cluster .dat files
    │   └── cluster_info_order_<N>/
    ├── hamiltonians_order_<N>/            # `*_site_info.dat`, `*_inter_all.dat`, ...
    │   └── cluster_<id>_order_<order>/
    ├── ed_results_order_<N>/              # per-cluster ED outputs
    │   └── cluster_<id>_order_<order>/output/ed_results.h5
    └── nlc_results_order_<N>/             # post-NLCE summation outputs
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:  # pragma: no cover - exercised at import time only
    HAS_H5PY = False

__all__ = [
    "HAS_H5PY",
    "ClusterEntry",
    "setup_logging",
    "check_gpu_available",
    "get_cluster_files",
    "get_num_sites",
    "count_sites_in_info_file",
    "load_thermo_dataset",
    "load_tpq_thermo_dataset",
]


@dataclass(frozen=True)
class ClusterEntry:
    """One discovered cluster on disk.

    Attributes:
        cluster_id: integer ID parsed from `cluster_<id>_order_<order>`.
        order: NLCE expansion order (sites *or* triangles depending on
            the lattice generator).
        path: absolute path to the cluster ``.dat`` file.
    """

    cluster_id: int
    order: int
    path: str

    def as_tuple(self) -> tuple[int, int, str]:
        """Compatibility tuple ``(cluster_id, order, path)`` for legacy callers."""
        return (self.cluster_id, self.order, self.path)


# ---------------------------------------------------------------------------
# Logging / GPU detection.
# ---------------------------------------------------------------------------


def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """Configure root logger to emit to both ``log_file`` and stderr.

    Idempotent: re-invocation re-installs the handlers (Python's
    ``logging.basicConfig`` is a no-op once handlers exist, so we
    clear and re-add explicitly).
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def check_gpu_available() -> bool:
    """Return True iff ``nvidia-smi`` reports at least one GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        gpu_names = result.stdout.strip().split("\n")
        return bool(gpu_names and gpu_names[0])
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Cluster-file discovery.
# ---------------------------------------------------------------------------


_CLUSTER_RE = re.compile(r"cluster_(\d+)_order_(\d+)")


def get_cluster_files(cluster_info_dir: str) -> list[ClusterEntry]:
    """Walk ``cluster_info_dir`` for ``cluster_<id>_order_<order>.dat`` files.

    Returns the list sorted by ``(cluster_id, order)`` so iteration
    order is reproducible across runs and across machines.
    """
    pattern = os.path.join(cluster_info_dir, "cluster_*_order_*.dat")
    out: list[ClusterEntry] = []
    for path in glob.glob(pattern):
        match = _CLUSTER_RE.search(os.path.basename(path))
        if match:
            out.append(ClusterEntry(
                cluster_id=int(match.group(1)),
                order=int(match.group(2)),
                path=path,
            ))
    out.sort(key=lambda e: (e.cluster_id, e.order))
    return out


def get_num_sites(cluster_info_path: str) -> Optional[int]:
    """Parse ``# Number of vertices: N`` out of a cluster info file."""
    try:
        with open(cluster_info_path, "r") as f:
            for line in f:
                if "Number of vertices:" in line:
                    return int(line.split(":")[1].strip())
    except OSError:
        return None
    return None


def count_sites_in_info_file(ham_subdir: str) -> Optional[int]:
    """Count non-comment, non-blank lines in ``<ham_subdir>/*_site_info.dat``."""
    matches = glob.glob(os.path.join(ham_subdir, "*_site_info.dat"))
    if not matches:
        return None
    n = 0
    with open(matches[0], "r") as f:
        for line in f:
            if not line.startswith("#") and line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# HDF5 / text-file output readers (used by the per-cluster plotting steps).
# ---------------------------------------------------------------------------


def load_thermo_dataset(output_dir: str) -> Optional[dict]:
    """Read the canonical thermodynamics curves from one ED run.

    Tries (in order):
      1. ``<output_dir>/ed_results.h5  ::  /thermodynamics/...``
      2. ``<output_dir>/ed_results.h5  ::  /ftlm/averaged/...``
      3. ``<output_dir>/thermo/thermo_data.txt``  (legacy whitespace format)

    Returns ``{T, energy, specific_heat, entropy, free_energy}`` (each
    value is a NumPy array or ``None``) on success, else ``None``.
    """
    import numpy as np

    h5_file = os.path.join(output_dir, "ed_results.h5")

    def _h5_pull(group_path: str) -> Optional[dict]:
        try:
            with h5py.File(h5_file, "r") as f:
                if group_path not in f:
                    return None
                grp = f[group_path]
                if "temperatures" not in grp:
                    return None
                return {
                    "T": grp["temperatures"][:],
                    "energy": grp["energy"][:] if "energy" in grp else None,
                    "specific_heat": grp["specific_heat"][:] if "specific_heat" in grp else None,
                    "entropy": grp["entropy"][:] if "entropy" in grp else None,
                    "free_energy": grp["free_energy"][:] if "free_energy" in grp else None,
                }
        except Exception as e:  # pragma: no cover
            logging.debug("HDF5 read failure on %s::%s: %s", h5_file, group_path, e)
            return None

    if HAS_H5PY and os.path.exists(h5_file):
        for group in ("/thermodynamics", "/ftlm/averaged"):
            data = _h5_pull(group)
            if data is not None:
                return data

    text_file = os.path.join(output_dir, "thermo", "thermo_data.txt")
    if os.path.exists(text_file):
        try:
            data = np.atleast_2d(np.loadtxt(text_file, comments="#"))
            return {
                "T": data[:, 0],
                "energy": data[:, 1] if data.shape[1] > 1 else None,
                "specific_heat": data[:, 2] if data.shape[1] > 2 else None,
                "entropy": data[:, 3] if data.shape[1] > 3 else None,
                "free_energy": data[:, 4] if data.shape[1] > 4 else None,
            }
        except Exception as e:
            logging.warning("Failed to parse %s: %s", text_file, e)
    return None


def load_tpq_thermo_dataset(output_dir: str) -> Optional[dict]:
    """Read TPQ-style thermodynamics (β, energy, variance) from one run.

    Tries (in order):
      1. ``<output_dir>/ed_results.h5  ::  /tpq/averaged/thermodynamics``
      2. ``<output_dir>/ed_results.h5  ::  /tpq/samples/sample_0/thermodynamics``
      3. ``<output_dir>/SS_rand0.dat``  (legacy text format)

    Returns ``{T, energy, specific_heat}`` on success, else ``None``.
    """
    import numpy as np

    def _from_table(table) -> dict:
        # Layout: (beta, energy, variance, doublon, step)
        betas = table[:, 0]
        return {
            "T": 1.0 / betas,
            "energy": table[:, 1],
            "specific_heat": table[:, 2] * betas ** 2,
        }

    h5_file = os.path.join(output_dir, "ed_results.h5")
    if HAS_H5PY and os.path.exists(h5_file):
        try:
            with h5py.File(h5_file, "r") as f:
                if "/tpq/averaged/thermodynamics" in f:
                    return _from_table(f["/tpq/averaged/thermodynamics"][:])
                if "/tpq/samples/sample_0/thermodynamics" in f:
                    return _from_table(f["/tpq/samples/sample_0/thermodynamics"][:])
        except Exception as e:  # pragma: no cover
            logging.debug("TPQ HDF5 read failure on %s: %s", h5_file, e)

    legacy = os.path.join(output_dir, "SS_rand0.dat")
    if os.path.exists(legacy):
        try:
            ss = np.loadtxt(legacy, unpack=True, skiprows=2)
            return {
                "T": 1.0 / ss[0],
                "energy": ss[1],
                "specific_heat": ss[2] * ss[0] ** 2,
            }
        except Exception as e:
            logging.warning("Failed to parse %s: %s", legacy, e)
    return None
