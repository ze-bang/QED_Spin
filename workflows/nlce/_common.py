"""
Shared infrastructure for the NLCE driver scripts.

This module is the *single source of truth* for the boilerplate that
the three driver scripts (`run/nlce.py`, `run/nlce_ftlm.py`,
`run/nlce_triangular.py`) used to copy-and-paste between themselves:

* logging setup (file + console),
* cluster-file discovery + parsing,
* Hilbert-space-aware ED method selection,
* the `subprocess.run(...)` retry / exit-code-vs-output reconciliation
  that swallows the (well-documented) "ED crashed during cleanup but
  the HDF5 file is intact" failure mode,
* HDF5/text-file thermodynamics readback with graceful fallbacks.

The drivers themselves stay focused on *workflow orchestration* (which
clusters, which model, which post-processing). All ED-CLI-touching
code lives here.

Layout convention (used by all three drivers):

    <base_dir>/
    ├── clusters_order_<N>/             # generated cluster .dat files
    │   └── cluster_info_order_<N>/
    ├── hamiltonians_order_<N>/         # per-cluster `*_site_info.dat` etc.
    │   └── cluster_<id>_order_<order>/
    ├── ed_results_order_<N>/           # per-cluster ED outputs
    │   └── cluster_<id>_order_<order>/output/ed_results.h5
    └── nlc_results_order_<N>/          # post-NLCE summation outputs

The ED CLI calls produced here always target the canonical `./ED`
binary built by the project (see `MODERNIZATION_AUDIT.md` P2.14).
"""

from __future__ import annotations

import glob
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable, Optional

try:
    import h5py
    HAS_H5PY = True
except ImportError:  # pragma: no cover - exercised at import time only
    HAS_H5PY = False

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


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
DEFAULT_ED_PATH = os.path.join(_WORKSPACE_ROOT, "build", "ED")


# ---------------------------------------------------------------------------
# Plain dataclasses for cluster entries and ED-runner options.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterEntry:
    """One discovered cluster on disk.

    Attributes:
        cluster_id: integer ID parsed out of `cluster_<id>_order_<order>`.
        order: NLCE expansion order (number of sites *or* triangles
            depending on the lattice generator).
        path: absolute path to the cluster `.dat` file.
    """

    cluster_id: int
    order: int
    path: str

    def as_tuple(self) -> tuple[int, int, str]:
        """Compatibility tuple `(cluster_id, order, path)` for legacy callers."""
        return (self.cluster_id, self.order, self.path)


@dataclass
class EDOptions:
    """Knobs the drivers want to forward into `./ED` invocations.

    Mirrors the union of every CLI flag the three drivers used to thread
    through their own ad-hoc dicts. Drivers populate the relevant fields
    and then hand the struct to `build_ed_command(...)`.
    """

    method: str = "FULL"  # FULL, SCALAPACK_MIXED, FULL_GPU, FTLM, FTLM_GPU, ...
    eigenvalues: Optional[str] = "FULL"
    spin_length: float = 0.5
    thermo: bool = False
    temp_min: float = 0.001
    temp_max: float = 20.0
    temp_bins: int = 100
    measure_spin: bool = False
    symmetrized: bool = False  # legacy --symmetrized; auto-translates from --symm
    use_symm: bool = True  # add `--symm` (auto-select symmetry mode) by default
    streaming_symmetry: bool = False
    basis_cache_dir: Optional[str] = None
    samples: Optional[int] = None  # FTLM samples
    krylov_dim: Optional[int] = None  # FTLM Krylov dimension
    extra_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Misc helpers (logging / GPU detection / cluster file walking).
# ---------------------------------------------------------------------------


def check_gpu_available() -> bool:
    """Return True iff `nvidia-smi` reports at least one GPU.

    Used by the drivers to decide whether `--method=FULL_GPU` /
    `--method=FTLM_GPU` is even worth attempting.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        gpu_names = result.stdout.strip().split("\n")
        return bool(gpu_names and gpu_names[0])
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def setup_logging(log_file: str, level: int = logging.INFO) -> None:
    """Configure root logger to emit to both `log_file` and stderr.

    Idempotent: re-invocation just re-installs the handlers (Python's
    `logging.basicConfig` is a no-op if the root already has handlers,
    so we clear and re-add explicitly).
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


_CLUSTER_RE = re.compile(r"cluster_(\d+)_order_(\d+)")


def get_cluster_files(cluster_info_dir: str) -> list[ClusterEntry]:
    """Walk `cluster_info_dir` for `cluster_<id>_order_<order>.dat` files.

    Returns the list sorted by `(cluster_id, order)` so iteration order
    is reproducible across runs (and across machines / glob orderings).
    """
    pattern = os.path.join(cluster_info_dir, "cluster_*_order_*.dat")
    out: list[ClusterEntry] = []
    for path in glob.glob(pattern):
        match = _CLUSTER_RE.search(os.path.basename(path))
        if match:
            out.append(
                ClusterEntry(
                    cluster_id=int(match.group(1)),
                    order=int(match.group(2)),
                    path=path,
                )
            )
    out.sort(key=lambda e: (e.cluster_id, e.order))
    return out


def get_num_sites(cluster_info_path: str) -> Optional[int]:
    """Parse `# Number of vertices: N` out of a cluster info file."""
    try:
        with open(cluster_info_path, "r") as f:
            for line in f:
                if "Number of vertices:" in line:
                    return int(line.split(":")[1].strip())
    except OSError:
        return None
    return None


def count_sites_in_info_file(ham_subdir: str) -> Optional[int]:
    """Count non-comment, non-blank lines in `<ham_subdir>/*_site_info.dat`.

    Used by every driver as a fallback for clusters whose generator
    didn't write `# Number of vertices:`.
    """
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
# ED-CLI command builder + subprocess driver.
# ---------------------------------------------------------------------------


def build_ed_command(
    ed_executable: str,
    ham_subdir: str,
    output_dir: str,
    num_sites: int,
    options: EDOptions,
    *,
    scalapack_threshold: int = 16,
    use_scalapack: bool = True,
) -> list[str]:
    """Translate `EDOptions` into a fully-formed `./ED` argv vector.

    Method selection rules (matching the drivers' historical behaviour):

      * `options.method == 'FULL_GPU'`  -> respected verbatim.
      * `options.method` is FTLM-flavoured -> respected verbatim.
      * `options.method == 'FULL'` (the default) and the cluster is large
        (`num_sites >= scalapack_threshold` AND `use_scalapack` is True)
        -> rewrite to `SCALAPACK_MIXED` for distributed diagonalization.
      * Otherwise, use `options.method` as-is.

    The Hilbert-space dimension is annotated in a debug log line so the
    NLCE workflow logs make method-selection decisions auditable.
    """
    method = options.method.upper()
    hilbert_dim = 2 ** num_sites

    if method == "FULL" and use_scalapack and num_sites >= scalapack_threshold:
        method = "SCALAPACK_MIXED"

    cmd: list[str] = [
        ed_executable,
        ham_subdir,
        f"--method={method}",
        f"--output={output_dir}/output",
        f"--num_sites={num_sites}",
        f"--spin_length={options.spin_length}",
    ]

    if options.eigenvalues:
        cmd.append(f"--eigenvalues={options.eigenvalues}")

    if options.symmetrized:
        cmd.append("--symmetrized")
    elif options.use_symm:
        cmd.append("--symm")

    if options.streaming_symmetry:
        cmd.append("--streaming-symmetry")
        if options.basis_cache_dir and os.path.isdir(options.basis_cache_dir):
            cmd.append(f"--basis-cache-dir={options.basis_cache_dir}")

    if options.measure_spin:
        cmd.append("--measure_spin")

    if options.samples is not None:
        cmd.append(f"--samples={options.samples}")
    if options.krylov_dim is not None:
        cmd.append(f"--krylov_dim={options.krylov_dim}")

    if options.thermo:
        cmd.extend([
            "--thermo",
            f"--temp_min={options.temp_min}",
            f"--temp_max={options.temp_max}",
            f"--temp_bins={options.temp_bins}",
        ])

    cmd.extend(options.extra_flags)

    logging.debug(
        "build_ed_command: cluster=%s sites=%d dim=2^%d=%d method=%s argv=%s",
        ham_subdir, num_sites, num_sites, hilbert_dim, method, cmd,
    )
    return cmd


def _output_dir_has_results(output_dir: str) -> bool:
    """Best-effort check that an HDF5 / legacy output file is salvageable.

    The ED binary occasionally crashes during MPI / GPU tear-down *after*
    successfully writing its HDF5 results file. This helper lets the
    drivers treat such crashes as success without losing the genuine
    failure mode (segfault that produces no output).
    """
    if not os.path.isdir(output_dir):
        return False
    h5_candidates = [
        os.path.join(output_dir, "ed_results.h5"),
        os.path.join(output_dir, "thermo", "ed_results.h5"),
    ]
    for path in h5_candidates:
        if os.path.exists(path):
            if HAS_H5PY:
                try:
                    with h5py.File(path, "r") as f:
                        if "eigenvalues" in f and f["eigenvalues"].shape[0] == 0:
                            return False
                    return True
                except Exception:  # pragma: no cover - corrupt file
                    return False
            return True
    txt_candidates = [
        os.path.join(output_dir, "thermo", "thermo_data.txt"),
        os.path.join(output_dir, "thermo", "ftlm_thermo.txt"),
    ]
    return any(os.path.exists(p) for p in txt_candidates)


def run_ed_subprocess(
    cmd: list[str],
    *,
    output_root: str,
    timeout: Optional[int] = None,
    extra_env: Optional[dict[str, str]] = None,
    log_tag: str = "ED",
) -> bool:
    """Run an `./ED` invocation and reconcile exit-code vs file output.

    Returns True if the run produced usable output (even when the
    binary itself exited non-zero), False otherwise. `output_root` is
    the directory passed via `--output=<output_root>/output` so we can
    inspect it on failure.

    `timeout` defaults to env var `NLCE_ED_TIMEOUT` (1 hr) when None.
    """
    env = os.environ.copy()
    env["ED_PYTHON"] = sys.executable
    if extra_env:
        env.update(extra_env)

    if timeout is None:
        try:
            timeout = int(os.environ.get("NLCE_ED_TIMEOUT", 3600))
        except (TypeError, ValueError):
            timeout = 3600

    output_dir = os.path.join(output_root, "output")

    try:
        subprocess.run(cmd, check=True, capture_output=True, env=env, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logging.error("[%s] subprocess timed out after %ds: %s", log_tag, timeout, cmd[0])
        return False
    except subprocess.CalledProcessError as e:
        if _output_dir_has_results(output_dir):
            logging.warning(
                "[%s] subprocess exited with code %d but usable output exists -- treating as success",
                log_tag, e.returncode,
            )
            return True

        if e.returncode == -11:  # SIGSEGV
            logging.error("[%s] subprocess crashed with SIGSEGV (no output produced)", log_tag)
        else:
            logging.error("[%s] subprocess failed: %s", log_tag, e)

        if e.stdout:
            stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else e.stdout
            logging.error("[%s] stdout: %s", log_tag, stdout)
        if e.stderr:
            stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else e.stderr
            logging.error("[%s] stderr: %s", log_tag, stderr)
        return False


# ---------------------------------------------------------------------------
# HDF5 / text-file output readers (used by the per-cluster plotting steps).
# ---------------------------------------------------------------------------


def load_thermo_dataset(output_dir: str) -> Optional[dict]:
    """Read the canonical thermodynamics curves from one ED run.

    Tries (in order):

      1. `<output_dir>/ed_results.h5  ::  /thermodynamics/...`
      2. `<output_dir>/ed_results.h5  ::  /ftlm/averaged/...`
      3. `<output_dir>/thermo/thermo_data.txt`  (legacy whitespace format)

    Returns `{T, energy, specific_heat, entropy, free_energy}` (each
    value is a NumPy array or None) on success, or None on failure.
    """
    import numpy as np  # local import keeps this module cheap to import

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
        except Exception as e:  # pragma: no cover - corrupt file
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

      1. `<output_dir>/ed_results.h5  ::  /tpq/averaged/thermodynamics`
      2. `<output_dir>/ed_results.h5  ::  /tpq/samples/sample_0/thermodynamics`
      3. `<output_dir>/SS_rand0.dat`  (legacy text format)

    Returns `{T, energy, specific_heat}` on success, None otherwise.
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
