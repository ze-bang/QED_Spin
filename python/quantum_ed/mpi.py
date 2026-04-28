"""Phase 5 / Phase 8: thin Python wrapper over the standalone
``ed_distributed_main`` binary.

The C++ MPI distributed solvers (``ed::distributed::distributed_lanczos``,
``distributed_ftlm``) are designed to be driven from an
``mpirun`` / ``mpiexec`` launcher on an HPC cluster. They are deliberately
**not** bound to ``quantum_ed._core``: a single-process Python interpreter
cannot host ``MPI_Init`` cleanly, and the right idiomatic launch is
``mpiexec -n N ed_distributed_main ...`` (or
``srun -n N ed_distributed_main ...`` on SLURM).

This module provides a tiny helper, :func:`run_distributed`, that builds
the right command-line and shells out for you so notebook callers don't
have to remember the launcher syntax.

Phase 8 fix: the wrapper now matches the *actual* CLI surface of the
``ed_distributed_main`` binary (``--mode lanczos|ftlm`` plus model
parameters). Earlier versions wrote ``--method=<m>`` and a leading
``directory`` argument which the binary never consumed -- the launches
silently ignored those tokens and ran the default Heisenberg chain
instead. The previous ``MPI_METHODS`` tuple also advertised three
methods (``tpq``, ``lanczos_symmetry``, ``lanczos_gpu``) that the
standalone driver does not expose; those are still available from
within Python via the :mod:`quantum_ed.distributed` extension module on
MPI-capable builds, but not via this subprocess shim.

Example
-------

.. code-block:: python

    from quantum_ed import mpi as qed_mpi

    # Lanczos on an N=20 Heisenberg chain:
    qed_mpi.run_distributed(
        method="lanczos",
        n_ranks=8,
        binary_args=("--N", "20", "--J", "1.0", "--max-iter", "200",
                     "--reorth", "1", "--periodic", "1", "--seed", "42"),
        launcher="srun",  # default is "mpiexec"
    )

For the exact CLI surface, run ``ed_distributed_main --help``.

The capability is also exposed through the build introspection helpers
in ``quantum_ed`` itself: ``quantum_ed.has_mpi_build()`` reports whether
the companion C++ build was made with ``WITH_MPI=ON`` (the precondition
for ``ed_distributed_main`` to exist).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from typing import Optional, Sequence

# The standalone ``ed_distributed_main`` driver exposes a fixed set of
# solver "modes" through its ``--mode`` flag. Bump this tuple in lockstep
# with the binary's parse_args() switch (``src/cli/ed_distributed_main.cpp``).
#
# Phase 9 (Apr 2026): added "tpq" → distributed_tpq (canonical TPQ across
# MPI ranks) and "krylov_schur" → distributed_krylov_schur (thick-restart
# Lanczos with locking). The Python wrapper accepts the kwarg
# unconditionally; the binary itself enforces availability based on its
# build flags.
MPI_METHODS = (
    "lanczos",       # distributed_lanczos          (CPU; GPU per rank if --gpu)
    "krylov_schur",  # distributed_krylov_schur     (thick-restart + locking)
    "ftlm",          # distributed_ftlm
    "tpq",           # distributed_tpq              (canonical TPQ)
)

# Solver names whose only distributed kernel is via the standalone binary
# above. ``run_distributed`` accepts these as ``method=`` for ergonomics
# (so users can copy-paste the ``solver=`` value from ``qed.diag``).
#
# Each entry maps ``alias -> (binary_mode, warn)``. ``warn=True`` means
# the alias represents a semantic downgrade (e.g. an alias mapping to a
# weaker kernel because the requested one has no distributed
# implementation) and we should emit a UserWarning. ``warn=False`` is
# for surface-equivalent renames (``mTPQ`` → ``tpq`` because
# ``distributed_tpq`` IS the canonical TPQ kernel; the case difference
# is just naming).
_METHOD_ALIASES = {
    "ks":             ("krylov_schur", False),
    "mtpq":           ("tpq",          False),
    "ctpq":           ("tpq",          False),
}


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
    method: str,
    n_ranks: int,
    *,
    use_gpu: bool = False,
    binary_args: Sequence[str] = (),
    launcher: str = "mpiexec",
    launcher_args: Sequence[str] = (),
    binary: Optional[str] = None,
    launcher_binary: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
    # ------------------------------------------------------------------
    # Deprecated arguments -- kept for one minor version so existing
    # call sites get a warning instead of a silent behavior change. The
    # ``directory`` positional was always a no-op (``ed_distributed_main``
    # never consumed it); ``extra_args`` was concatenated *after*
    # ``--method=...``, which the binary also never consumed.
    # ------------------------------------------------------------------
    directory: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> subprocess.CompletedProcess:
    """Launch ``mpiexec -n N ed_distributed_main --mode <method> ...`` and wait.

    Parameters
    ----------
    method : str
        One of :data:`MPI_METHODS` (``"lanczos"``, ``"ftlm"``, ``"tpq"``).
        Common aliases from ``qed.diag(solver=...)`` are also accepted:
        ``"ks"`` maps to ``"krylov_schur"`` (real distributed kernel
        since Phase 9), and ``"mtpq"`` / ``"ctpq"`` map to ``"tpq"``.
    n_ranks : int
        Number of MPI ranks. Forwarded to the launcher as ``-n N``.
    use_gpu : bool, optional
        If True, append ``--gpu`` to the binary command. The
        ``ed_distributed_main`` driver routes
        ``--mode lanczos --gpu`` through ``distributed_lanczos_gpu``
        (multi-GPU per rank + NCCL allreduce) and
        ``--mode tpq --gpu`` through ``distributed_tpq_gpu``
        (Phase 9 / Layer 2: device-resident |psi> + cuBLAS axpys/
        dotcs + NCCL allreduces, MPI-over-samples). The binary
        errors cleanly if it was built without ``WITH_CUDA=ON``;
        this Python wrapper forwards the flag blindly because the
        binary's build flags can't be introspected without
        launching it.
    binary_args : sequence of str, optional
        Extra CLI flags forwarded to ``ed_distributed_main`` after the
        ``--mode`` token; e.g. ``("--N", "20", "--max-iter", "400",
        "--reorth", "1")``. See ``ed_distributed_main --help`` for the
        full surface (``--N``, ``--J``, ``--periodic``, ``--max-iter``,
        ``--exct``, ``--reorth``, ``--seed``, ``--samples``,
        ``--groups``, ``--betas``, ``--verbose``).
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
    env : dict, optional
        Environment overrides for the subprocess.
    check : bool, optional
        If True (default), raise ``CalledProcessError`` on non-zero exit.
    capture_output : bool, optional
        If True, capture stdout/stderr in the returned object.
    directory : str, optional
        **Deprecated.** Pre-Phase-8 versions accepted a directory
        positional that was silently ignored by the binary. Passing it
        now raises a ``DeprecationWarning`` and the value is dropped.
    extra_args : sequence of str, optional
        **Deprecated.** Use ``binary_args`` instead. If provided, the
        contents are appended to ``binary_args`` and a
        ``DeprecationWarning`` is emitted.

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
    # Accept the same names users pass to qed.diag(solver=...) and map
    # them onto the binary's --mode tokens. Case-insensitive.
    requested = method
    method_key = method.lower()
    if method_key in _METHOD_ALIASES:
        aliased, should_warn = _METHOD_ALIASES[method_key]
        if should_warn:
            warnings.warn(
                f"run_distributed(method={method!r}) maps to "
                f"--mode {aliased!r} on ed_distributed_main "
                f"(no distributed kernel exists for {method!r} itself; "
                "the closest equivalent is being used).",
                stacklevel=2,
            )
        method = aliased
    if method not in MPI_METHODS:
        raise ValueError(
            f"method={requested!r} not in {MPI_METHODS}. "
            "ed_distributed_main exposes a fixed set of MPI solver modes; "
            "extend MPI_METHODS in quantum_ed/mpi.py if you add a new one. "
            "See qed.solver_device_support() for the full (solver, "
            "device) compatibility matrix."
        )

    if directory is not None:
        warnings.warn(
            "quantum_ed.mpi.run_distributed(directory=...) is deprecated and "
            "ignored: ed_distributed_main never consumed a directory "
            "positional argument. Drop the directory= kwarg from your call.",
            DeprecationWarning,
            stacklevel=2,
        )
    if extra_args is not None:
        warnings.warn(
            "quantum_ed.mpi.run_distributed(extra_args=...) is deprecated; "
            "use binary_args= instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        binary_args = tuple(binary_args) + tuple(extra_args)

    launcher_path = _resolve_binary(launcher, launcher_binary)
    binary_path = _resolve_binary("ed_distributed_main", binary)

    # The binary uses `--mode <name>` (two tokens), not `--method=<name>`.
    # ``--gpu`` is the (Phase 9) flag that switches the inner kernel to
    # the GPU variant (e.g. distributed_lanczos_gpu). The binary errors
    # cleanly if it was built without WITH_CUDA=ON; we forward the flag
    # blindly because we can't introspect the binary's build flags
    # from Python without launching it.
    gpu_flag = ("--gpu",) if use_gpu else ()
    cmd = [
        launcher_path, "-n", str(int(n_ranks)),
        *launcher_args,
        binary_path,
        "--mode", method,
        *gpu_flag,
        *binary_args,
    ]
    return subprocess.run(
        cmd,
        check=check,
        env=env,
        capture_output=capture_output,
        text=True,
    )


__all__ = ["MPI_METHODS", "run_distributed"]
