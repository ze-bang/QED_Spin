"""``qed._solve.device``: the ``device=`` picker and the capability matrix.

Carved out of ``workflow.py`` (WP11). Answers two questions: which
``(use_gpu, use_mpi)`` pair a ``device=`` request resolves to, and which
``(solver, device)`` cells the C++ side has kernels for / this build can
actually reach.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .. import _core as _core


# ---------------------------------------------------------------------------
# Build-flag probes. ``has_cuda_build`` / ``has_mpi_build`` come from
# ``qed._core``, but they were module globals of ``workflow.py`` before this
# split and faking a CUDA build by patching them THERE
# (``monkeypatch.setattr(qed.workflow, "has_cuda_build", ...)``) is the seam
# the device gates use. Importing the ``_core`` functions straight into this
# module would freeze the unpatched versions into it and silently ignore such
# a patch, so the probes are read off the facade at call time instead -- with
# nothing patched they resolve to the very same ``_core`` functions. (The
# facade imports these names from ``qed._core``, never from here, so the
# delegation cannot recurse.)
# ---------------------------------------------------------------------------

def has_cuda_build() -> bool:
    from .. import workflow as _facade
    return bool(_facade.has_cuda_build())


def has_mpi_build() -> bool:
    from .. import workflow as _facade
    return bool(_facade.has_mpi_build())


# ---------------------------------------------------------------------------
# Solver x Device compatibility introspection
#
# Static metadata (which solvers have which device kernels) plus a
# build-aware "is this actually reachable on the current build?" check.
# ---------------------------------------------------------------------------

# Per (solver_family, device) cell: True if the C++ side has a kernel
# wired for that combination, False if there's no such kernel.
# "device" axis values:
#   "cpu"     -> single-process CPU
#   "gpu"     -> single GPU (cuSPARSE / per-sector dispatch)
#   "mpi"     -> distributed via mpirun on the CLI `ED` binary (the
#                ed_distributed_main binary was retired in Stage 11d;
#                the in-process qed.solve surface raises for 'mpi')
#   "mpi_gpu" -> distributed CPU + per-rank GPU (multi-GPU)
#
# Coverage refreshed 2026-07-30 against src/orchestrator.cpp dispatch.
_SOLVER_DEVICE_KERNELS: dict[str, dict[str, bool]] = {
    "LANCZOS":         {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "BLOCK_LANCZOS":   {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "KRYLOV_SCHUR":    {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "FULL":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "BLOCK_KRYLOV_SCHUR": {"cpu": True, "gpu": True, "mpi": False, "mpi_gpu": False},
    "mTPQ":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "FTLM":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "OFTLM":           {"cpu": True, "gpu": False, "mpi": False, "mpi_gpu": False},
    # Audit 2026-07-30: LTLM and KPM_DOS dispatch on CudaBackend in the
    # orchestrator (src/orchestrator.cpp LTLM/KpmDos lanes + kpm_dos_gpu.cu),
    # and the capability matrix publishes passing gpu rows for both -- the
    # old False entries contradicted the shipped kernels. OFTLM remains the
    # sole CPU-only thermal lane (H.bind_cpu() in its orchestrator branch).
    "LTLM":            {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
    "KPM_DOS":         {"cpu": True, "gpu": True,  "mpi": False, "mpi_gpu": False},
}


def solver_device_support(
    *,
    solver: Optional[str] = None,
    return_dict: bool = False,
) -> Optional[dict[str, dict[str, dict[str, Any]]]]:
    """Inspect which (solver, device) cells are reachable on this build.

    The compatibility matrix has two layers:

    * **kernel** -- whether a C++ kernel exists for the cell at all
      (set at compile time of the C++ side).
    * **build** -- whether THIS python build can reach it
      (``WITH_CUDA`` / ``WITH_MPI`` / ``WITH_NCCL`` flags). For
      example, the LANCZOS-on-GPU cell exists in C++, but a build
      with ``WITH_CUDA=OFF`` cannot reach it.

    Without a ``solver`` argument the function prints a table; pass
    ``return_dict=True`` to get the matrix back as nested dicts of
    ``{solver: {device: {"kernel": bool, "available": bool, "note":
    str}}}``.

    Parameters
    ----------
    solver : str, optional
        Filter to one solver family (e.g. ``"LANCZOS"``,
        ``"KRYLOV_SCHUR"``, ``"mTPQ"``). Substring match is allowed.
    return_dict : bool, optional
        If True, suppress printing and return the structured matrix.

    Returns
    -------
    dict or None
        When ``return_dict=True``, the nested support matrix.

    Notes
    -----
    The MPI subprocess cells were retired (the ed_distributed_main launcher
    and qed.mpi are gone). What remains is in the CLI: run `ED` under mpirun
    and each rank solves a disjoint subset of the symmetry sectors, with the
    spectrum Allgatherv-d at the end. No in-process lane builds MpiBackend --
    select_backend picks it only for a distributed geometry, and every basis
    policy reports is_distributed() == false.
    """
    cuda_ok = bool(has_cuda_build())
    mpi_ok = bool(has_mpi_build())

    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for solver_name, devices in _SOLVER_DEVICE_KERNELS.items():
        if solver is not None and solver.upper() not in solver_name.upper():
            continue
        cells: dict[str, dict[str, Any]] = {}
        for device, has_kernel in devices.items():
            if not has_kernel:
                cells[device] = {
                    "kernel": False,
                    "available": False,
                    "note": "no C++ kernel for this combination",
                }
                continue
            if device == "cpu":
                cells[device] = {"kernel": True, "available": True, "note": ""}
            elif device == "gpu":
                cells[device] = {
                    "kernel": True,
                    "available": cuda_ok,
                    "note": ("" if cuda_ok
                             else "build has WITH_CUDA=OFF; rebuild with "
                                  "-DWITH_CUDA=ON"),
                }
            elif device in ("mpi", "mpi_gpu"):
                cells[device] = {
                    "kernel": False,
                    "available": False,
                    "note": ("retired: run the CLI under mpirun -- each rank "
                             "takes a disjoint set of symmetry sectors"),
                }
        matrix[solver_name] = cells

    if return_dict:
        return matrix

    print(f"Build flags: WITH_CUDA={'ON' if cuda_ok else 'OFF'}, "
          f"WITH_MPI={'ON' if mpi_ok else 'OFF'}")
    print()
    devices = ["cpu", "gpu", "mpi", "mpi_gpu"]
    header = f"{'solver':<22}" + "".join(f"{d:>11}" for d in devices)
    print(header)
    print("-" * len(header))
    for solver_name, cells in matrix.items():
        row = f"{solver_name:<22}"
        for d in devices:
            cell = cells[d]
            if not cell["kernel"]:
                row += f"{'-':>11}"
            elif cell["available"]:
                row += f"{'OK':>11}"
            else:
                row += f"{'(unbuilt)':>11}"
        print(row)
    print()
    print("Legend:  OK = wired and reachable on this build;")
    print("         (unbuilt) = C++ kernel exists but this build is missing")
    print("                     the WITH_CUDA / WITH_MPI flag;")
    print("         -  = no C++ kernel for this (solver, device) combination.")
    return None


def _resolve_device(device: Optional[str], dim: int) -> tuple[bool, bool]:
    """Pick (use_gpu, use_mpi).

    Single-GPU is honoured for any solver the in-process build supports,
    via a temp-dir routing in :func:`solve` (see ``_diag_via_directory``).
    ``device='mpi'``/``'mpi_gpu'`` (the retired subprocess launcher)
    raise with guidance.

    Returns ``(use_gpu, use_mpi)``; ``use_mpi`` is always False now.
    """
    if device is None or device == "auto":
        # GPU audit (2026-09-11): the CudaBackend lane overtakes the 16-thread
        # CPU lane at dim ~ 7e5 (N = 22 half filling) for the Krylov verbs and
        # is 1.5-3x SLOWER below ~2e5 (launch + sync latency per iteration), so
        # "auto" keeps dim < 2^18 on the CPU. Pass device="gpu" to force it.
        # ... and only when a device is actually THERE: a CUDA build with no usable
        # GPU (empty allocation, broken card, driver older than the toolkit) would
        # otherwise pick the GPU lane and be served by the host without a word.
        use_gpu = bool(has_cuda_build()) and bool(_core.have_cuda()) and dim >= (1 << 18)
        return use_gpu, False
    device_lc = device.lower()
    if device_lc == "cpu":
        return False, False
    if device_lc == "gpu":
        if not has_cuda_build():
            raise RuntimeError(
                "device='gpu' requested but this build of qed._core "
                "does not have WITH_CUDA=ON. Rebuild with -DWITH_CUDA=ON or "
                "use device='cpu'."
            )
        # A CUDA BUILD is not a usable DEVICE. Without this check the GPU lanes
        # quietly run on the host when the allocation has no GPU, the device is
        # broken or the driver is older than the build's toolkit -- a node with a
        # failed GPU once produced 19 "GPU" golden results computed on the CPU.
        if not _core.have_cuda():
            raise RuntimeError(
                "device='gpu' requested but no usable CUDA device is visible to "
                f"this process on {os.uname().nodename} (cudaGetDeviceCount found "
                "none: no GPU in the allocation, a broken or busy device, or a "
                "driver older than the toolkit this build used). The GPU lanes "
                "would silently run on the CPU; fix the allocation or pass "
                "device='cpu'."
            )
        return True, False
    if device_lc in ("mpi", "mpi_gpu"):
        raise RuntimeError(
            "device='mpi' / 'mpi_gpu' was retired: the subprocess launcher "
            "(ed_distributed_main + qed.mpi) and the distributed-operator "
            "family behind it were removed. For MPI, run the CLI under "
            "mpirun: each rank solves a disjoint subset of the symmetry "
            "sectors and the spectrum is Allgatherv-d. No in-process lane "
            "builds MpiBackend (select_backend needs a distributed geometry "
            "and no basis policy produces one). Single-node frontier runs "
            "use device='gpu' (fp32 mTPQ / rep-lane memory scaling)."
        )
    raise ValueError(
        f"device={device!r} not in "
        "{'auto', 'cpu', 'gpu', 'mpi', 'mpi_gpu'}."
    )
