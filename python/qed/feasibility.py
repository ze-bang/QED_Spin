# =============================================================================
# python/qed/feasibility.py    (Phase 9 / Layer 6)
#
# Pre-flight planner for `qed.diag` / `qed.find_symmetries` /
# `qed.mpi.run_distributed`.
#
# Goal: given a Hamiltonian + a desired (solver, device, basis), tell the
# user **before** the kernel runs:
#
#   * what the basis dimension actually is (full / sz / symm / sym+sz),
#   * how much RAM / VRAM the chosen path will need (per-rank + total),
#   * how much wall-time it will roughly take,
#   * whether it fits on the host (or how many MPI ranks it needs to fit),
#   * a ranked list of cheaper alternatives if the request is infeasible.
#
# All numbers are rough engineering estimates -- they're meant to flag
# "this will OOM" or "this will take days" before you find out the hard
# way, not to predict the exact wall time. The host probe is best-effort
# (psutil + `nvidia-smi` if available) and falls back to env vars
# (`QED_HOST_MEMORY_GB`, `QED_HOST_GPU_MEMORY_GB`, `QED_HOST_N_GPUS`,
# `QED_HOST_N_MPI_RANKS`) if the system probe is unavailable.
# =============================================================================

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np

from . import _core
from ._core import (
    DiagonalizationMethod,
    FixedSzOperator,
    Operator,
)


# ---------------------------------------------------------------------------
# Constants tuned from the bench_vs_xdiag bake-off (Phase 6) and the
# DistributedOperator stage-3 NCCL/cuBLAS micro-benchmarks (Phase 3c).
# These are deliberately conservative: the planner exists to avoid OOM,
# not to be a perf oracle.
# ---------------------------------------------------------------------------

#: bytes per std::complex<double> element
BYTES_PER_COMPLEX = 16

#: nanoseconds per (Hamiltonian-term, basis-element) for a CPU SoA SpMV.
#: AVX2 + 32-thread Zen4 measured ~50 ns; we round up to 100 to cover
#: cache-cold and term-heavy spin models.
SPMV_NS_PER_TERM_ELEMENT_CPU = 100.0

#: nanoseconds per (Hamiltonian-term, basis-element) for the CUDA SoA
#: SpMV kernel from `gpu_operator.cu`. Measured ~1-3 ns/element on a
#: single H100 in the Phase 3c benchmarks; we round up to 5 to cover
#: smaller GPUs and term-heavy interactions.
SPMV_NS_PER_TERM_ELEMENT_GPU = 5.0

#: AllReduce latency floor (microseconds) for a single double across an
#: MPI / NCCL ring. A pessimistic 50 us lets us flag "thousands of dot
#: products on a tiny problem will be latency-bound".
ALLREDUCE_LATENCY_US = 50.0

#: Per-process Python interpreter + libraries baseline (GB). Used to
#: warn when the planned vectors fit "memory-wise" but the residual
#: budget for the OS / runtime is tight.
HOST_BASELINE_OVERHEAD_GB = 2.0


# =============================================================================
# Host probing
# =============================================================================

@dataclass
class HostResources:
    """Best-effort snapshot of the local host's compute resources.

    Attributes
    ----------
    cpu_memory_gb : float
        Total physical RAM (GiB). Falls back to 16 GB if undetected.
    gpu_memory_gb : float
        Per-device VRAM (GiB) for the *smallest* visible GPU (0 if none).
    n_gpus : int
        Number of CUDA-visible GPUs (0 if none).
    n_mpi_ranks_avail : int
        Best guess at number of MPI ranks the user could launch on this
        host (defaults to logical CPU count if no scheduler hints).
    has_cuda_build : bool
        ``qed._core.has_cuda_build()``.
    has_mpi_build : bool
        ``qed._core.has_mpi_build()``.
    has_nccl_build : bool
        Whether the C++ side was built with NCCL (multi-GPU collectives).
    notes : list[str]
        Diagnostic notes about how each field was determined (for the
        verbose planner output).
    """

    cpu_memory_gb: float = 16.0
    gpu_memory_gb: float = 0.0
    n_gpus: int = 0
    n_mpi_ranks_avail: int = 1
    has_cuda_build: bool = False
    has_mpi_build: bool = False
    has_nccl_build: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Host resources (best effort):",
            f"  cpu_memory       = {self.cpu_memory_gb:.1f} GB",
            f"  gpu_memory       = {self.gpu_memory_gb:.1f} GB / device "
            f"(n_gpus={self.n_gpus})",
            f"  n_mpi_ranks_avail= {self.n_mpi_ranks_avail}",
            f"  build flags      = WITH_CUDA={self.has_cuda_build}  "
            f"WITH_MPI={self.has_mpi_build}  "
            f"WITH_NCCL={self.has_nccl_build}",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)


def _probe_cpu_memory_gb() -> tuple[float, str]:
    # Prefer env var override, then psutil, then /proc/meminfo, then guess.
    env = os.environ.get("QED_HOST_MEMORY_GB")
    if env:
        try:
            return float(env), f"from env QED_HOST_MEMORY_GB={env}"
        except ValueError:
            pass
    try:
        import psutil  # type: ignore  # noqa: WPS433
        return psutil.virtual_memory().total / (1024.0 ** 3), "via psutil"
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 ** 2), "via /proc/meminfo"
    except OSError:
        pass
    return 16.0, "unknown (defaulted to 16 GB; set QED_HOST_MEMORY_GB to override)"


def _probe_gpu(has_cuda_build: bool) -> tuple[int, float, str]:
    env_n = os.environ.get("QED_HOST_N_GPUS")
    env_mem = os.environ.get("QED_HOST_GPU_MEMORY_GB")
    if env_n is not None or env_mem is not None:
        n = int(env_n) if env_n else 0
        mem = float(env_mem) if env_mem else 0.0
        return n, mem, "from env QED_HOST_N_GPUS / QED_HOST_GPU_MEMORY_GB"
    if not has_cuda_build:
        return 0, 0.0, "no CUDA build"
    nv = shutil.which("nvidia-smi")
    if nv is None:
        return 0, 0.0, "nvidia-smi not on PATH"
    try:
        out = subprocess.run(
            [nv, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return 0, 0.0, "nvidia-smi probe failed"
    mibs = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            mibs.append(int(line))
        except ValueError:
            pass
    if not mibs:
        return 0, 0.0, "nvidia-smi returned no devices"
    smallest_gb = min(mibs) / 1024.0
    return len(mibs), smallest_gb, f"via nvidia-smi ({len(mibs)} device(s))"


def _probe_n_mpi_ranks(cpu_count: int) -> tuple[int, str]:
    env = os.environ.get("QED_HOST_N_MPI_RANKS")
    if env:
        try:
            return int(env), f"from env QED_HOST_N_MPI_RANKS={env}"
        except ValueError:
            pass
    # Common scheduler env vars
    for var in ("SLURM_NTASKS", "PBS_NP", "OMPI_COMM_WORLD_SIZE"):
        v = os.environ.get(var)
        if v:
            try:
                return int(v), f"from env {var}={v}"
            except ValueError:
                pass
    return max(1, cpu_count), f"defaulted to logical CPU count ({cpu_count})"


def probe_host(*, cached: bool = True) -> HostResources:
    """Probe the local host for CPU RAM / GPU VRAM / MPI rank count.

    Cached on first call (use ``cached=False`` to refresh). Best effort:
    falls back to environment variables (``QED_HOST_*``) and finally to
    conservative defaults if every probe fails. The returned object
    carries ``notes`` describing how each field was determined.
    """
    global _HOST_CACHE
    if cached and _HOST_CACHE is not None:
        return _HOST_CACHE

    has_cuda = bool(getattr(_core, "has_cuda_build", lambda: False)())
    has_mpi = bool(getattr(_core, "has_mpi_build", lambda: False)())
    has_nccl = bool(getattr(_core, "has_nccl_build", lambda: False)()) \
        if hasattr(_core, "has_nccl_build") else has_cuda and has_mpi

    cpu_count = os.cpu_count() or 1
    cpu_mem, cpu_note = _probe_cpu_memory_gb()
    n_gpus, gpu_mem, gpu_note = _probe_gpu(has_cuda)
    n_ranks, ranks_note = _probe_n_mpi_ranks(cpu_count)

    h = HostResources(
        cpu_memory_gb=cpu_mem,
        gpu_memory_gb=gpu_mem,
        n_gpus=n_gpus,
        n_mpi_ranks_avail=n_ranks,
        has_cuda_build=has_cuda,
        has_mpi_build=has_mpi,
        has_nccl_build=bool(has_nccl),
        notes=[
            f"cpu_memory: {cpu_note}",
            f"gpu: {gpu_note}",
            f"n_mpi_ranks_avail: {ranks_note}",
        ],
    )
    _HOST_CACHE = h
    return h


_HOST_CACHE: Optional[HostResources] = None


# =============================================================================
# Operator + basis introspection
# =============================================================================

@dataclass
class OperatorMetadata:
    """Cheap, no-IO snapshot of a Hamiltonian's structure."""

    num_sites: int
    dim_full: int
    n_one_body: int
    n_two_body: int
    n_three_body: int
    conserves_sz: bool
    is_fixed_sz: bool
    fixed_sz_dim: Optional[int]  # if is_fixed_sz, the C(N, n_up) sector dim

    @property
    def n_terms(self) -> int:
        return self.n_one_body + self.n_two_body + self.n_three_body

    @property
    def has_three_body(self) -> bool:
        return self.n_three_body > 0


def inspect_operator(H: Union[Operator, FixedSzOperator]) -> OperatorMetadata:
    """Cheap structural inspection: term counts, dim, Sz conservation."""
    is_fsz = isinstance(H, FixedSzOperator)
    base: Operator = H  # FixedSzOperator inherits Operator

    n1 = len(base.iter_one_body_terms())
    n2 = len(base.iter_two_body_terms())
    try:
        n3 = len(base.iter_three_body_terms())
    except Exception:
        n3 = 0
    try:
        conserves = bool(base.conserves_sz())
    except Exception:
        conserves = False

    fsz_dim: Optional[int] = None
    if is_fsz:
        try:
            fsz_dim = int(H.dimension)
        except Exception:
            fsz_dim = None

    return OperatorMetadata(
        num_sites=int(base.num_sites),
        dim_full=int(1 << int(base.num_sites)),
        n_one_body=n1,
        n_two_body=n2,
        n_three_body=n3,
        conserves_sz=conserves,
        is_fixed_sz=is_fsz,
        fixed_sz_dim=fsz_dim,
    )


@dataclass
class BasisChoice:
    """One of the four ways to constrain the working basis."""

    kind: str           # "full" | "sz" | "symm" | "sym+sz"
    dim: int
    description: str

    def __repr__(self) -> str:
        return f"BasisChoice(kind={self.kind!r}, dim={self.dim:_d})"


def _sz_dim(num_sites: int, n_up: int) -> int:
    if not (0 <= n_up <= num_sites):
        raise ValueError(f"n_up={n_up} out of range [0, {num_sites}]")
    return math.comb(num_sites, n_up)


def _symmetry_group_size(symmetry: Any) -> int:
    """Best-effort group-size lookup for the planner's dim estimate."""
    if symmetry is None:
        return 1
    # GeneratorSet (workflow.GeneratorSet) -- has group_size attr.
    if hasattr(symmetry, "group_size"):
        try:
            return max(1, int(symmetry.group_size))
        except Exception:
            return 1
    # dict produced by group_from_generators
    if isinstance(symmetry, dict):
        n = symmetry.get("group_size") or symmetry.get("n_elements")
        try:
            return max(1, int(n))
        except Exception:
            return 1
    if isinstance(symmetry, (list, tuple)):
        # raw list of permutations; take 2^len as a *cap*, fall back to len+1
        return max(2, len(symmetry) + 1)
    return 1


def enumerate_basis(
    meta: OperatorMetadata,
    *,
    sz: Optional[int] = None,
    symmetry: Any = None,
    sector_size_estimate: Optional[int] = None,
) -> BasisChoice:
    """Pick the basis the workflow will actually run on, given (sz, symmetry)."""
    N = meta.num_sites
    if meta.is_fixed_sz:
        dim = meta.fixed_sz_dim or _sz_dim(N, N // 2)
        if symmetry is not None:
            G = _symmetry_group_size(symmetry)
            sym_dim = sector_size_estimate or max(1, dim // max(1, G))
            return BasisChoice(
                kind="sym+sz",
                dim=int(sym_dim),
                description=(
                    f"FixedSzOperator(N={N}, dim={dim:_d}) projected onto a "
                    f"symmetry irrep of group_size={G} "
                    f"=> {sym_dim:_d}"
                ),
            )
        return BasisChoice(
            kind="sz",
            dim=int(dim),
            description=f"FixedSzOperator(N={N}) sector dim={dim:_d}",
        )

    if sz is not None:
        if not meta.conserves_sz:
            raise ValueError(
                f"sz={sz} requested but operator does not conserve total Sz; "
                "drop sz= or check the Hamiltonian.")
        dim = _sz_dim(N, int(sz))
        if symmetry is not None:
            G = _symmetry_group_size(symmetry)
            sym_dim = sector_size_estimate or max(1, dim // max(1, G))
            return BasisChoice(
                kind="sym+sz",
                dim=int(sym_dim),
                description=(
                    f"sz=n_up={sz} (dim={dim:_d}) projected onto a "
                    f"symmetry irrep of group_size={G} "
                    f"=> {sym_dim:_d}"
                ),
            )
        return BasisChoice(
            kind="sz",
            dim=int(dim),
            description=f"sz=n_up={sz} sector dim={dim:_d}",
        )

    if symmetry is not None:
        G = _symmetry_group_size(symmetry)
        sym_dim = sector_size_estimate or max(1, meta.dim_full // max(1, G))
        return BasisChoice(
            kind="symm",
            dim=int(sym_dim),
            description=(
                f"Full Hilbert (dim={meta.dim_full:_d}) projected onto a "
                f"symmetry irrep of group_size={G} "
                f"=> {sym_dim:_d}"
            ),
        )

    return BasisChoice(
        kind="full",
        dim=int(meta.dim_full),
        description=f"Full Hilbert space dim=2^{N}={meta.dim_full:_d}",
    )


# =============================================================================
# Solver memory + time models
# =============================================================================

def _normalize_method_name(method: Union[str, DiagonalizationMethod]) -> str:
    if isinstance(method, DiagonalizationMethod):
        return method.name.upper()
    return str(method).upper()


def _vector_count_for(
    method_name: str,
    *,
    num_eigenvalues: int,
    max_subspace: Optional[int],
    n_samples: Optional[int],
    compute_eigenvectors: bool,
    block_size: Optional[int],
) -> tuple[int, str]:
    """Number of dim-sized complex vectors the chosen kernel keeps live.

    Returns (count, rationale). ``count`` is the **resident** working set;
    sequential samples (TPQ / FTLM outer loop) are NOT multiplied in.
    """
    name = method_name.upper()
    nev = max(1, num_eigenvalues)
    ms = max_subspace if (max_subspace and max_subspace > 0) else \
        max(80, 4 * nev + 40)
    bs = max(1, int(block_size or 1))

    # Eigenvalue solvers ----------------------------------------------------
    if "FULL" in name and "SCALAPACK" not in name:
        # Dense H + dense eigvecs: 2 * dim^2 doubles. We model this as
        # "vector count = dim" so the planner sees dim * dim memory.
        return -1, "FULL dense LAPACK: ~2 * dim^2 complex<double>"
    if "SCALAPACK" in name:
        return -1, "SCALAPACK dense: ~2 * dim^2 complex<double> distributed"
    if name.startswith("ARPACK"):
        ncv = max(2 * nev + 1, ms)
        return ncv + 3, f"ARPACK NCV={ncv} resident Arnoldi vectors + 3 scratch"
    if "DAVIDSON" in name or "LOBPCG" in name:
        return ms * 2 + 4, f"DAVIDSON/LOBPCG basis V + AV (~2*max_subspace={ms})"
    if "BLOCK_KRYLOV_SCHUR" in name:
        return ms + bs * 2 + 3, (
            f"BLOCK_KRYLOV_SCHUR: max_subspace={ms} + 2*block_size={bs}")
    if "KRYLOV_SCHUR" in name:
        if compute_eigenvectors:
            return ms + 4, f"KRYLOV_SCHUR: thick-restart basis (~max_subspace={ms})"
        return 4, "KRYLOV_SCHUR: 3 working vectors (basis regenerated)"
    if "BLOCK_LANCZOS" in name:
        return ms + bs * 2 + 3, (
            f"BLOCK_LANCZOS: max_subspace={ms} + 2*block_size={bs}")
    if "LANCZOS" in name:
        if compute_eigenvectors:
            return ms + 4, (
                f"LANCZOS w/ eigenvectors: stored basis (~max_subspace={ms})")
        return 4, "LANCZOS: ~4 working vectors (regenerated 2-pass)"
    if "SHIFT_INVERT" in name or "BICG" in name or "OSS" in name \
            or "CHEBYSHEV" in name:
        return ms + 6, (
            f"{name}: ~max_subspace={ms} + scratch for the linear solve")

    # Thermal solvers (sequential outer loop -> no n_samples multiplier) ----
    if "TPQ" in name:
        return 5, "TPQ: |psi> + term + Hterm + result + Hpsi (one sample at a time)"
    if "FTLM" in name or "LTLM" in name or "HYBRID" in name:
        # Inner Lanczos basis (regen 2-pass) + observable scratch.
        return ms + 4, (
            f"FTLM/LTLM: inner Lanczos basis (~max_subspace={ms}) + observable scratch")

    # Unknown -> conservative default
    return ms + 4, f"unknown solver: defaulting to max_subspace={ms} + 4"


def estimate_memory_gb(
    dim: int,
    method: Union[str, DiagonalizationMethod],
    *,
    num_eigenvalues: int = 1,
    max_subspace: Optional[int] = None,
    n_samples: Optional[int] = None,
    block_size: Optional[int] = None,
    compute_eigenvectors: bool = False,
    n_ranks: int = 1,
) -> tuple[float, float, str]:
    """Per-rank and total resident memory (GB) for the chosen kernel.

    Returns ``(per_rank_gb, total_gb, rationale)``. The model treats the
    Hilbert-space basis as a balanced 1-D slab across ``n_ranks``;
    matrix-free SpMV doesn't store H, so memory == vectors.
    """
    method_name = _normalize_method_name(method)
    vec_count, rationale = _vector_count_for(
        method_name,
        num_eigenvalues=num_eigenvalues,
        max_subspace=max_subspace,
        n_samples=n_samples,
        compute_eigenvectors=compute_eigenvectors,
        block_size=block_size,
    )
    n_ranks = max(1, int(n_ranks))

    if vec_count < 0:
        # Dense path: memory is dim * dim, distributed for SCALAPACK.
        bytes_total = 2.0 * (dim ** 2) * BYTES_PER_COMPLEX
        per_rank = bytes_total / n_ranks
        return (
            per_rank / (1024.0 ** 3),
            bytes_total / (1024.0 ** 3),
            f"{rationale}; total = 2 * {dim}^2 * 16 B "
            f"= {bytes_total / (1024.0 ** 3):.1f} GB",
        )

    # Iterative path: vectors are slabbed across ranks.
    total_bytes = float(vec_count) * float(dim) * BYTES_PER_COMPLEX
    per_rank = total_bytes / n_ranks
    return (
        per_rank / (1024.0 ** 3),
        total_bytes / (1024.0 ** 3),
        f"{rationale}; per vector = {dim:_d} * 16 B = "
        f"{(dim * BYTES_PER_COMPLEX) / (1024.0 ** 3):.2f} GB; "
        f"resident vectors = {vec_count}",
    )


def estimate_time_s(
    dim: int,
    n_terms: int,
    method: Union[str, DiagonalizationMethod],
    *,
    num_eigenvalues: int = 1,
    max_iterations: Optional[int] = None,
    n_samples: int = 1,
    block_size: Optional[int] = None,
    device: str = "cpu",
    n_ranks: int = 1,
    n_gpus_per_rank: int = 1,
) -> tuple[float, str]:
    """Very rough wall-time estimate (seconds). Returns (seconds, rationale).

    Model: time = (cost_per_spmv) * (num_spmv_total) where
    ``cost_per_spmv ≈ n_terms * (dim / n_workers) * NS_PER_TERM_ELEMENT``.
    The number of SpMVs is solver-dependent (heuristic table).
    """
    method_name = _normalize_method_name(method)
    nev = max(1, num_eigenvalues)
    max_iter = max_iterations if (max_iterations and max_iterations > 0) \
        else max(200, 8 * nev + 80)
    n_samples = max(1, int(n_samples))
    n_ranks = max(1, int(n_ranks))

    # SpMV count per call (number of H * v inside the kernel).
    if "FULL" in method_name and "SCALAPACK" not in method_name:
        # LAPACK zheev: ~ (8/3) * dim^3 flops; rough wall-time at 100 GFLOPs
        flops = (8.0 / 3.0) * (dim ** 3)
        seconds = flops / (100.0e9)  # 100 GFLOPS dense BLAS rule of thumb
        return seconds, (
            f"FULL: ~(8/3) dim^3 flops at ~100 GFLOPS dense = "
            f"{seconds:.2g} s")
    if "SCALAPACK" in method_name:
        flops = (8.0 / 3.0) * (dim ** 3)
        seconds = flops / (100.0e9 * n_ranks)
        return seconds, (
            f"SCALAPACK: ~(8/3) dim^3 flops distributed across {n_ranks} ranks "
            f"@ 100 GFLOPS each = {seconds:.2g} s")

    if "TPQ" in method_name:
        # n_samples * n_betas * delta_betas/step * taylor_order
        n_spmv_per_sample = max_iter
        n_spmv = n_samples * n_spmv_per_sample
        rationale_extra = (
            f"TPQ: {n_samples} sample(s) * ~{n_spmv_per_sample} SpMV (delta-beta * Taylor)")
    elif "FTLM" in method_name or "LTLM" in method_name:
        # Two-pass Lanczos (basis + reproject) per sample
        n_spmv_per_sample = 2 * max_iter
        n_spmv = n_samples * n_spmv_per_sample
        rationale_extra = (
            f"FTLM/LTLM: {n_samples} sample(s) * 2 * {max_iter} = {n_spmv} SpMV")
    elif "BLOCK" in method_name:
        bs = max(1, int(block_size or nev))
        n_spmv = max_iter * bs
        rationale_extra = (
            f"BLOCK_*: {max_iter} iter * block_size={bs} = {n_spmv} SpMV")
    elif "ARPACK" in method_name:
        # Implicitly restarted Arnoldi: ncv * (max_restarts+1)
        ncv = 2 * nev + 1
        n_spmv = ncv * 4  # 4 implicit restarts is generous default
        rationale_extra = f"ARPACK: NCV={ncv} * 4 restarts ~ {n_spmv} SpMV"
    elif "DAVIDSON" in method_name or "LOBPCG" in method_name:
        n_spmv = max_iter * max(1, nev)
        rationale_extra = (
            f"DAVIDSON/LOBPCG: {max_iter} iter * nev={nev} = {n_spmv} SpMV")
    else:
        # Lanczos / Krylov-Schur family: 2-pass when eigenvectors are kept;
        # we conservatively use 2x.
        n_spmv = 2 * max_iter
        rationale_extra = (
            f"LANCZOS/KRYLOV_SCHUR: 2 * {max_iter} = {n_spmv} SpMV")

    if device in ("gpu", "mpi_gpu"):
        ns_per_elem = SPMV_NS_PER_TERM_ELEMENT_GPU
        # mpi_gpu: each rank holds dim / n_ranks elements
        per_rank_dim = dim // max(1, n_ranks)
        per_spmv_s = n_terms * per_rank_dim * ns_per_elem * 1e-9
        device_label = f"GPU ({SPMV_NS_PER_TERM_ELEMENT_GPU} ns/elem)"
    else:
        ns_per_elem = SPMV_NS_PER_TERM_ELEMENT_CPU
        per_rank_dim = dim // max(1, n_ranks)
        per_spmv_s = n_terms * per_rank_dim * ns_per_elem * 1e-9
        device_label = f"CPU ({SPMV_NS_PER_TERM_ELEMENT_CPU} ns/elem)"

    seconds = n_spmv * per_spmv_s
    rationale = (
        f"{rationale_extra} on {device_label} with n_terms={n_terms} and "
        f"per-rank dim={per_rank_dim:_d}: total ~{seconds:.2g} s")
    return seconds, rationale


# =============================================================================
# Feasibility report + estimator
# =============================================================================

@dataclass
class FeasibilityReport:
    """Pre-flight verdict for one (solver, device, basis, n_ranks) plan."""

    feasible: bool
    bottleneck: str  # "memory" | "time" | "kernel" | "build" | "ok"
    basis: BasisChoice
    solver_name: str
    device: str
    n_ranks: int
    memory_per_rank_gb: float
    total_memory_gb: float
    available_per_rank_gb: float
    time_estimate_s: float
    notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    @property
    def fits_in_one_rank(self) -> bool:
        return self.memory_per_rank_gb <= self.available_per_rank_gb

    def summary(self) -> str:
        verdict = "FEASIBLE" if self.feasible else f"INFEASIBLE ({self.bottleneck})"
        lines = [
            f"[qed.diag.planner] verdict: {verdict}",
            f"  basis     : {self.basis.description}",
            f"  solver    : {self.solver_name}",
            f"  device    : {self.device}  (n_ranks={self.n_ranks})",
            f"  memory    : per-rank {self.memory_per_rank_gb:.2f} GB / "
            f"avail {self.available_per_rank_gb:.2f} GB    "
            f"(total {self.total_memory_gb:.2f} GB across {self.n_ranks} "
            f"rank{'s' if self.n_ranks != 1 else ''})",
            f"  wall-time : ~{_fmt_time(self.time_estimate_s)}  (rough order of magnitude)",
        ]
        for n in self.notes:
            lines.append(f"  note: {n}")
        if self.suggestions:
            lines.append("  suggestions:")
            for s in self.suggestions:
                lines.append(f"    - {s}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _fmt_time(s: float) -> str:
    if s < 0.1:
        return f"{s * 1000.0:.1f} ms"
    if s < 60:
        return f"{s:.2f} s"
    if s < 3600:
        return f"{s / 60.0:.2f} min"
    if s < 86400:
        return f"{s / 3600.0:.2f} h"
    return f"{s / 86400.0:.2f} d"


class ResourceError(RuntimeError):
    """Raised when the planner judges the requested workflow infeasible.

    Carries the :class:`FeasibilityReport` (incl. ``suggestions``) on the
    ``report`` attribute so callers can introspect.
    """

    def __init__(self, message: str, report: "FeasibilityReport"):
        super().__init__(message)
        self.report = report


def estimate_resources(
    H: Union[Operator, FixedSzOperator],
    *,
    solver: Union[str, DiagonalizationMethod] = "LANCZOS",
    device: str = "cpu",
    sz: Optional[int] = None,
    symmetry: Any = None,
    num_eigenvalues: int = 1,
    n_samples: Optional[int] = None,
    n_ranks: Optional[int] = None,
    max_subspace: Optional[int] = None,
    max_iterations: Optional[int] = None,
    block_size: Optional[int] = None,
    compute_eigenvectors: bool = False,
    sector_size_estimate: Optional[int] = None,
    host: Optional[HostResources] = None,
) -> FeasibilityReport:
    """Plan one (solver, device, basis, n_ranks) combination.

    Parameters mirror :func:`qed.diag` so the planner can be
    invoked with the same kwargs the user is about to dispatch with.
    Returns a :class:`FeasibilityReport`. Does **not** raise on
    infeasibility; the caller decides whether to abort or proceed.
    """
    host = host or probe_host()
    meta = inspect_operator(H)
    basis = enumerate_basis(meta, sz=sz, symmetry=symmetry,
                             sector_size_estimate=sector_size_estimate)

    method_name = _normalize_method_name(solver)
    notes: list[str] = []
    suggestions: list[str] = []

    # ------------------------------------------------------------------
    # Resolve n_ranks for the chosen device.
    # ------------------------------------------------------------------
    device_lc = device.lower()
    if device_lc in ("mpi", "mpi_gpu"):
        nr = int(n_ranks) if n_ranks else max(1, host.n_mpi_ranks_avail)
    elif device_lc == "gpu":
        nr = 1  # single-GPU path
    else:
        nr = 1
    nr = max(1, nr)

    # ------------------------------------------------------------------
    # Memory + time estimates.
    # ------------------------------------------------------------------
    per_rank_gb, total_gb, mem_rationale = estimate_memory_gb(
        basis.dim, solver,
        num_eigenvalues=num_eigenvalues,
        max_subspace=max_subspace,
        n_samples=n_samples,
        block_size=block_size,
        compute_eigenvectors=compute_eigenvectors,
        n_ranks=nr,
    )
    seconds, time_rationale = estimate_time_s(
        basis.dim, max(1, meta.n_terms), solver,
        num_eigenvalues=num_eigenvalues,
        max_iterations=max_iterations,
        n_samples=n_samples or 1,
        block_size=block_size,
        device=device_lc,
        n_ranks=nr,
        n_gpus_per_rank=1,
    )
    notes.append(mem_rationale)
    notes.append(time_rationale)

    # ------------------------------------------------------------------
    # Available memory (per rank, on this host).
    # ------------------------------------------------------------------
    if device_lc in ("gpu", "mpi_gpu"):
        avail = max(0.0, host.gpu_memory_gb - 0.5)
        if device_lc == "gpu" and host.n_gpus < 1:
            notes.append("device='gpu' requested but no CUDA-visible GPU detected")
        if device_lc == "mpi_gpu":
            avail = max(0.0, host.gpu_memory_gb - 0.5)
            if host.n_gpus < 1:
                notes.append("device='mpi_gpu' requested but no GPU detected on this host")
    else:
        # CPU / MPI: per-rank budget = (cpu_memory - baseline) / n_ranks_local
        ranks_local = nr if device_lc == "mpi" else 1
        avail = max(0.0,
                     (host.cpu_memory_gb - HOST_BASELINE_OVERHEAD_GB) /
                     max(1, ranks_local))

    # ------------------------------------------------------------------
    # Build / kernel availability checks.
    # ------------------------------------------------------------------
    bottleneck = "ok"
    feasible = True
    if device_lc in ("gpu", "mpi_gpu") and not host.has_cuda_build:
        feasible = False
        bottleneck = "build"
        suggestions.append(
            "rebuild with -DWITH_CUDA=ON, or pass device='cpu'/'mpi'.")
    if device_lc in ("mpi", "mpi_gpu") and not host.has_mpi_build:
        notes.append(
            "Python extension was built without WITH_MPI; the workflow will "
            "still try to spawn ed_distributed_main externally.")
    if device_lc == "mpi_gpu" and not host.has_nccl_build:
        notes.append(
            "WITH_NCCL is off in this build; multi-GPU collectives will fail "
            "at runtime (rebuild with NCCL_FOUND, or use device='mpi').")

    if "TPQ" in method_name and (basis.kind in ("symm", "sym+sz")) \
            and basis.kind == "symm" \
            and device_lc not in ("mpi", "mpi_gpu"):
        # In-process streaming kernel falls back to Lanczos for
        # TPQ + symm. The distributed path (Phase E) DOES support
        # per-sector canonical TPQ via `distributed_tpq_symmetry` /
        # `distributed_tpq_gpu_symmetry`; for those device cells
        # the combination is feasible (caller aggregates across
        # sectors when reconstructing full-space Z).
        feasible = False
        bottleneck = "kernel"
        suggestions.append(
            "TPQ on a symmetry-projected irrep falls back to Lanczos in "
            "the in-process streaming kernel. Drop symmetry=, pre-project "
            "to a fixed-Sz block, or use device='mpi'/'mpi_gpu' (Phase E "
            "wires per-sector canonical TPQ; caller aggregates over "
            "sectors)."
        )

    # ------------------------------------------------------------------
    # Memory verdict.
    #
    # Trip the memory bottleneck when the per-rank vector working set
    # exceeds the available budget OR when the budget is non-positive
    # but the request would actually allocate something (>1 MB).
    # ------------------------------------------------------------------
    if feasible and (
        per_rank_gb > avail
        or (avail <= 0.0 and per_rank_gb > 1e-3)
    ):
        feasible = False
        bottleneck = "memory"
        # Heuristic suggestions ranked by how much they reduce dim:
        if not meta.is_fixed_sz and meta.conserves_sz and sz is None:
            half = meta.num_sites // 2
            sz_dim = _sz_dim(meta.num_sites, half)
            ratio = max(1.0, basis.dim / max(1, sz_dim))
            suggestions.append(
                f"pass sz={half} -- dim {basis.dim:_d} -> {sz_dim:_d} "
                f"(~{ratio:.1f}x smaller)")
        if symmetry is None:
            suggestions.append(
                "pass symmetry=qed.find_symmetries(H).full_set to project "
                "onto the largest commuting subgroup (typically 4x-32x for "
                "lattice models)")
        if device_lc in ("cpu", "gpu"):
            extra_ranks = math.ceil(per_rank_gb / max(1e-6, avail))
            suggestions.append(
                f"switch to device='mpi' with mpi_n_ranks>={extra_ranks} "
                f"(or device='mpi_gpu' if a multi-GPU build is available)")
        if "FULL" in method_name and basis.dim > 1024:
            suggestions.append(
                "the FULL dense path scales as dim^2; switch to "
                "solver='LANCZOS' (or KRYLOV_SCHUR for many eigenvalues)")
        if compute_eigenvectors and basis.dim > 1_000_000:
            suggestions.append(
                "drop compute_eigenvectors=True (or write per-rank slabs "
                "via output_dir=...) -- storing full eigenvectors at this "
                "size dominates memory")

    # Cost suggestions (informational; don't flip feasible).
    if feasible and seconds > 24 * 3600:
        suggestions.append(
            f"estimated wall-time is {_fmt_time(seconds)}; consider raising "
            f"n_ranks, projecting onto a smaller sector (sz= / symmetry=), "
            f"or switching to a thermal solver (FTLM/TPQ) if you only need "
            f"thermodynamic averages")

    if not feasible and not suggestions:
        suggestions.append(
            "no obvious mitigation -- check the rationale above and "
            "consider a coarser tolerance or fewer eigenvalues")

    return FeasibilityReport(
        feasible=feasible,
        bottleneck=bottleneck,
        basis=basis,
        solver_name=method_name,
        device=device_lc,
        n_ranks=nr,
        memory_per_rank_gb=per_rank_gb,
        total_memory_gb=total_gb,
        available_per_rank_gb=avail,
        time_estimate_s=seconds,
        notes=notes,
        suggestions=suggestions,
    )


# =============================================================================
# Workflow suggestion engine: rank candidate (basis, solver, device) plans
# =============================================================================

@dataclass
class WorkflowCandidate:
    rationale: str
    solver: str
    device: str
    n_ranks: int
    sz: Optional[int]
    use_symmetry: bool
    estimated: FeasibilityReport

    def call_signature(self) -> str:
        kwargs: list[str] = []
        kwargs.append(f"solver={self.solver!r}")
        kwargs.append(f"device={self.device!r}")
        if self.device in ("mpi", "mpi_gpu"):
            kwargs.append(f"mpi_n_ranks={self.n_ranks}")
        if self.sz is not None:
            kwargs.append(f"sz={self.sz}")
        if self.use_symmetry:
            kwargs.append("symmetry=qed.find_symmetries(H).full_set")
        return f"qed.diag(H, {', '.join(kwargs)})"


@dataclass
class WorkflowSuggestion:
    intent: str
    operator: OperatorMetadata
    host: HostResources
    candidates: list[WorkflowCandidate]

    def best(self) -> Optional[WorkflowCandidate]:
        feasible = [c for c in self.candidates if c.estimated.feasible]
        if not feasible:
            return None
        # Rank by (per-rank memory headroom OK, then by time, then by total memory).
        feasible.sort(key=lambda c: (
            c.estimated.time_estimate_s,
            c.estimated.memory_per_rank_gb,
        ))
        return feasible[0]

    def summary(self, top: int = 5) -> str:
        lines = [
            "qed.suggest_workflow:",
            f"  intent           = {self.intent}",
            f"  operator         = N={self.operator.num_sites}, "
            f"dim_full=2^{self.operator.num_sites}={self.operator.dim_full:_d}, "
            f"n_terms={self.operator.n_terms}, "
            f"conserves_sz={self.operator.conserves_sz}",
            "",
            self.host.summary(),
            "",
            "Candidates (ranked, best first):",
        ]
        feasible = [c for c in self.candidates if c.estimated.feasible]
        infeasible = [c for c in self.candidates if not c.estimated.feasible]
        feasible.sort(key=lambda c: (
            c.estimated.time_estimate_s,
            c.estimated.memory_per_rank_gb,
        ))
        ranked = feasible + infeasible
        for i, c in enumerate(ranked[:top], 1):
            verdict = "OK" if c.estimated.feasible \
                else f"INFEASIBLE ({c.estimated.bottleneck})"
            lines.append(
                f"  [{i}] {verdict}  {c.rationale}")
            lines.append(f"      {c.call_signature()}")
            lines.append(
                f"      memory ~ {c.estimated.memory_per_rank_gb:.2f} GB/rank, "
                f"wall-time ~ {_fmt_time(c.estimated.time_estimate_s)}")
        if not ranked:
            lines.append("  (no candidates -- nothing to do)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _intent_solver(intent: str) -> str:
    intent = intent.lower()
    if intent in ("ground_state", "gs", "ground"):
        return "LANCZOS"
    if intent in ("low_lying", "low-lying", "spectrum"):
        return "KRYLOV_SCHUR"
    if intent in ("thermal", "thermo", "tpq", "ftlm", "finite_temperature"):
        return "FTLM"
    if intent in ("spectral", "dynamical", "dssf"):
        return "LANCZOS"  # spectral functions ride on Lanczos+continued-fraction
    raise ValueError(
        f"unknown intent={intent!r}; choose 'ground_state' / 'low_lying' / "
        "'thermal' / 'spectral'")


def suggest_workflow(
    H: Union[Operator, FixedSzOperator],
    *,
    intent: str = "ground_state",
    num_eigenvalues: int = 1,
    n_samples: Optional[int] = None,
    sz_candidates: Optional[Sequence[int]] = None,
    symmetry: Any = None,
    host: Optional[HostResources] = None,
    available_devices: Optional[Sequence[str]] = None,
) -> WorkflowSuggestion:
    """Produce a ranked list of (basis, solver, device) plans for a goal.

    Parameters
    ----------
    H : Operator or FixedSzOperator
    intent : str
        ``'ground_state'`` (default), ``'low_lying'``, ``'thermal'``,
        or ``'spectral'``. Selects the solver family the planner tries.
    num_eigenvalues : int
        Forwarded to the per-candidate planner.
    n_samples : int, optional
        Only used for thermal intent (TPQ/FTLM sample count).
    sz_candidates : sequence of int, optional
        Sz sectors to try (in ``n_up`` units). Defaults to
        ``[N//2]`` when the operator conserves Sz, ``[]`` otherwise.
    symmetry : optional
        If given, the planner will also include a (sym) plan and a
        (sym + sz) plan for each Sz candidate.
    host : HostResources, optional
        Override the auto-probed host (useful for "what would this
        cost on cluster X?" planning). Defaults to ``probe_host()``.
    available_devices : sequence of str, optional
        Subset of ``{'cpu', 'gpu', 'mpi', 'mpi_gpu'}`` to consider.
        Defaults to everything the build supports.
    """
    host = host or probe_host()
    meta = inspect_operator(H)
    base_solver = _intent_solver(intent)

    devs = list(available_devices) if available_devices else []
    if not devs:
        devs = ["cpu"]
        if host.has_cuda_build and host.n_gpus > 0:
            devs.append("gpu")
        if host.has_mpi_build or shutil.which("ed_distributed_main"):
            devs.append("mpi")
        if host.has_nccl_build and host.n_gpus > 0:
            devs.append("mpi_gpu")

    if sz_candidates is None:
        if meta.conserves_sz and not meta.is_fixed_sz:
            sz_candidates = [meta.num_sites // 2]
        else:
            sz_candidates = []

    # Build the cross-product (basis x device).
    basis_options: list[tuple[Optional[int], Any, str]] = [(None, None, "full Hilbert")]
    for s in sz_candidates:
        basis_options.append((int(s), None, f"sz={s}"))
    if symmetry is not None:
        basis_options.append((None, symmetry, "symmetry-projected"))
        for s in sz_candidates:
            basis_options.append((int(s), symmetry, f"sz={s} + symmetry"))

    candidates: list[WorkflowCandidate] = []
    for sz_v, sym_v, basis_label in basis_options:
        for dev in devs:
            try:
                rep = estimate_resources(
                    H,
                    solver=base_solver,
                    device=dev,
                    sz=sz_v,
                    symmetry=sym_v,
                    num_eigenvalues=num_eigenvalues,
                    n_samples=n_samples,
                    n_ranks=(host.n_mpi_ranks_avail
                             if dev in ("mpi", "mpi_gpu") else 1),
                    host=host,
                )
            except ValueError:
                continue
            candidates.append(WorkflowCandidate(
                rationale=f"{basis_label} on {dev}",
                solver=base_solver,
                device=dev,
                n_ranks=rep.n_ranks,
                sz=sz_v,
                use_symmetry=sym_v is not None,
                estimated=rep,
            ))

    return WorkflowSuggestion(
        intent=intent,
        operator=meta,
        host=host,
        candidates=candidates,
    )


__all__ = [
    "BYTES_PER_COMPLEX",
    "BasisChoice",
    "FeasibilityReport",
    "HostResources",
    "OperatorMetadata",
    "ResourceError",
    "WorkflowCandidate",
    "WorkflowSuggestion",
    "enumerate_basis",
    "estimate_memory_gb",
    "estimate_resources",
    "estimate_time_s",
    "inspect_operator",
    "probe_host",
    "suggest_workflow",
]
