"""Phase 5: thin Python wrapper over the standalone ``ed_distributed_main`` binary.

The C++ MPI distributed solvers (``ed::distributed::distributed_lanczos``,
``distributed_ftlm``, ``distributed_tpq``, ``distributed_lanczos_symmetry``,
``distributed_lanczos_gpu``) are designed to be driven from an
``mpirun`` / ``mpiexec`` launcher on an HPC cluster. They are deliberately
**not** bound to ``quantum_ed._core``: a single-process Python interpreter
cannot host ``MPI_Init`` cleanly, and the right idiomatic launch is
``mpiexec -n N ed_distributed_main ...`` (or
``srun -n N ed_distributed_main ...`` on SLURM).

This module provides a tiny helper, :func:`run_distributed`, that builds
the right command-line and shells out for you so notebook callers don't
have to remember the launcher syntax. For the exact CLI surface of
``ed_distributed_main`` (``--method=lanczos|ftlm|tpq|lanczos_symmetry|
lanczos_gpu``, ``--max-iter``, ``--tol``, ``--reorth``, etc.), see the
self-documenting ``--help`` of the binary.

Example
-------

.. code-block:: python

    from quantum_ed import mpi as qed_mpi

    qed_mpi.run_distributed(
        directory="/scratch/runs/heisenberg-32-chain",
        method="lanczos",
        n_ranks=8,
        extra_args=("--max-iter", "400", "--reorth", "1"),
        launcher="srun",  # default is "mpiexec"
    )

The capability is also exposed through the build introspection helpers in
``quantum_ed`` itself: ``quantum_ed.has_mpi_build()`` reports whether the
companion C++ build was made with ``WITH_MPI=ON`` (the precondition for
``ed_distributed_main`` to exist), and ``quantum_ed.has_cuda_build()``
reports whether ``--method=lanczos_gpu`` (NCCL halo path) will be
available in that binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, Sequence

# Mirror the four MPI methods ed_distributed_main exposes as of Phase 3c.
MPI_METHODS = (
    "lanczos",            # distributed_lanczos
    "ftlm",               # distributed_ftlm
    "tpq",                # distributed_tpq (canonical)
    "lanczos_symmetry",   # distributed_lanczos_symmetry
    "lanczos_gpu",        # distributed_lanczos_gpu (NCCL halo)
)


def _resolve_binary(name: str, override: Optional[str]) -> str:
    """Locate an executable, preferring an explicit override path."""
    if override:
        if not os.path.isfile(override):
            raise FileNotFoundError(
                f"{name} override path {override!r} does not exist"
            )
        return override
    on_path = shutil.which(name)
    if on_path is not None:
        return on_path
    raise FileNotFoundError(
        f"Could not find `{name}` on $PATH. Either build it (cmake "
        f"--build <build> --target {name}), put the build directory on "
        f"$PATH, or pass the {name}_binary= argument to run_distributed()."
    )


def run_distributed(
    directory: str,
    method: str,
    n_ranks: int,
    *,
    launcher: str = "mpiexec",
    launcher_args: Sequence[str] = (),
    binary: Optional[str] = None,
    launcher_binary: Optional[str] = None,
    extra_args: Sequence[str] = (),
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Launch ``mpiexec -n N ed_distributed_main ...`` and wait.

    Parameters
    ----------
    directory : str
        Path to the run directory the binary should consume (positional
        first argument to ``ed_distributed_main``).
    method : str
        One of ``"lanczos"``, ``"ftlm"``, ``"tpq"``,
        ``"lanczos_symmetry"``, or ``"lanczos_gpu"`` -- mapped to
        ``--method=<method>``.
    n_ranks : int
        Number of MPI ranks. Forwarded to the launcher as ``-n N``.
    launcher : str, optional
        Launcher executable name, default ``"mpiexec"``. Set to
        ``"srun"`` on SLURM, ``"mpirun"`` for OpenMPI users who prefer
        that name, etc.
    launcher_args : sequence of str, optional
        Extra arguments inserted between ``-n N`` and the binary. Useful
        for ``--bind-to=core`` or scheduler hints.
    binary : str, optional
        Absolute path to ``ed_distributed_main``. Defaults to ``shutil.which``.
    launcher_binary : str, optional
        Absolute path to the launcher; same default rule.
    extra_args : sequence of str, optional
        Extra CLI flags forwarded to ``ed_distributed_main`` (after
        ``--method=...``); e.g. ``("--max-iter", "400", "--reorth", "1")``.
    env : dict, optional
        Environment overrides for the subprocess.
    check : bool, optional
        If True (default), raise ``CalledProcessError`` on non-zero exit.
    capture_output : bool, optional
        If True, capture stdout/stderr in the returned object.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    ValueError
        If ``method`` is not one of :data:`MPI_METHODS`.
    FileNotFoundError
        If the launcher or ``ed_distributed_main`` cannot be found.
    """
    if method not in MPI_METHODS:
        raise ValueError(
            f"method={method!r} not in {MPI_METHODS}. "
            "ed_distributed_main exposes a fixed set of MPI solvers; "
            "extend MPI_METHODS in quantum_ed/mpi.py if you add a new one."
        )
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"directory={directory!r} does not exist or is not a directory"
        )

    launcher_path = _resolve_binary(launcher, launcher_binary)
    binary_path = _resolve_binary("ed_distributed_main", binary)

    cmd = [
        launcher_path, "-n", str(int(n_ranks)),
        *launcher_args,
        binary_path,
        directory,
        f"--method={method}",
        *extra_args,
    ]
    return subprocess.run(
        cmd,
        check=check,
        env=env,
        capture_output=capture_output,
        text=True,
    )


__all__ = ["MPI_METHODS", "run_distributed"]
