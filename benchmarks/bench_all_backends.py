#!/usr/bin/env python3
"""
NOTE (Stage 11d, Jul 2026): the ed_distributed_main launcher was
retired; the --mpi-ranks / --gpu-mpi-ranks lanes of this script are
non-functional and skip unless the binary is restored from git history.

benchmarks/bench_all_backends.py

Unified head-to-head benchmark driver.

For each backend that is built into the local CMake tree, this script
times two canonical kernels on a 1D Heisenberg PBC chain across a range
of system sizes:

  * Kernel A -- one matrix-vector product H @ v on a unit-norm complex
    state vector (lower bound on every iterative solver's per-iter cost).
  * Kernel B -- one full ground-state Lanczos to numerical convergence
    (lower bound on a real ED workflow's wall time).

Backends covered:
  * cpu_complex   : Operator::apply complex path (this repo, OpenMP).
  * cpu_real      : Operator::apply_real real path (this repo, OpenMP).
  * gpu           : GPUOperator::matVecGPU / GPULanczos (cuSPARSE + cuBLAS).
  * distributed   : ed::distributed::DistributedOperator + distributed_lanczos
                    via mpiexec -n {1,2,4,8} on the local node.
  * distributed_gpu : ed::distributed::distributed_lanczos_gpu (cuBLAS local +
                    NCCL collectives), driven by `ed_distributed_main --gpu`
                    via mpiexec, sweep over --gpu-mpi-ranks (skipped if no
                    NCCL / no GPU).
  * quspin        : QuSpin's hamiltonian + scipy.sparse.linalg.eigsh.
  * scipy_csr     : QuSpin's CSR matrix + numpy @-product (apply only;
                    Lanczos shares the eigsh backend with QuSpin so we
                    do not double-count).

Output:
  * STDOUT pretty-printed table grouped by backend.
  * `BENCHMARKS_OUT/bench_all_backends.json` -- single JSON artefact
    used by `docs/benchmarks/BENCHMARKS.md` for the cross-platform plots.

Usage:
  python3 benchmarks/bench_all_backends.py                  \
      --build-dir build                                     \
      --sizes 12 14 16 18 20                                \
      --threads $(nproc)                                    \
      --mpi-ranks 1 2 4                                     \
      --output bench_all_backends.json

This script ASSUMES:
  - The CMake build at --build-dir already contains:
       build/benchmarks/bench_operator_apply
       build/benchmarks/bench_lanczos_ground_state
       build/benchmarks/bench_gpu_operator_apply           (if WITH_CUDA)
       build/benchmarks/bench_gpu_lanczos_ground_state     (if WITH_CUDA)
       build/ed_distributed_main                           (if WITH_MPI)
  - quspin, scipy, numpy are importable.
  - mpiexec is on $PATH (skipped gracefully if not).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


# ============================================================================
# C++ Google-Benchmark helpers
# ============================================================================

def gbench_run(binary: str, filter_re: str, env: dict[str, str],
               min_time: str = "0.4s") -> list[dict[str, Any]]:
    """Invoke a Google-Benchmark binary and parse its JSON output."""
    if not os.path.exists(binary):
        return []
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        out_path = f.name
    cmd = [
        binary,
        f"--benchmark_out={out_path}",
        "--benchmark_out_format=json",
        f"--benchmark_filter={filter_re}",
        f"--benchmark_min_time={min_time}",
    ]
    full_env = {**os.environ, **env}
    try:
        subprocess.run(cmd, env=full_env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    try:
        with open(out_path) as fh:
            j = json.load(fh)
    except Exception:
        return []
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass
    return j.get("benchmarks", [])


def gbench_time_for_size(bench_list: list[dict[str, Any]], N: int,
                         name_prefix: str) -> float | None:
    """Return microseconds for the bench whose name == f"{name_prefix}/{N}"."""
    target = f"{name_prefix}/{N}"
    for b in bench_list:
        if b["name"].split("/min_time")[0] == target:
            return float(b["real_time"])
    return None


def gbench_time_for_lanczos(bench_list: list[dict[str, Any]], N: int) -> float | None:
    """Return microseconds for BM_LanczosGroundState/{N}/<krylov> entries."""
    for b in bench_list:
        parts = b["name"].split("/")
        if len(parts) >= 2 and parts[0] in ("BM_LanczosGroundState",
                                             "BM_GPULanczosGroundState") \
                and parts[1] == str(N):
            return float(b["real_time"])
    return None


# ============================================================================
# QuSpin / scipy reference implementations
# ============================================================================

def quspin_chain(N: int):
    """Heisenberg J=1 chain, PBC, no symmetries."""
    from quspin.basis import spin_basis_general
    from quspin.operators import hamiltonian

    basis = spin_basis_general(N=N, S="1/2", pauli=False)
    Jzz = [[1.0, i, (i + 1) % N] for i in range(N)]
    Jpm = [[0.5, i, (i + 1) % N] for i in range(N)]
    static = [["zz", Jzz], ["+-", Jpm], ["-+", Jpm]]
    return hamiltonian(static, [], basis=basis,
                       dtype=np.float64,
                       check_symm=False, check_herm=False, check_pcon=False)


def time_quspin_apply(H, n_warmup: int = 1, n_calls: int = 5) -> float:
    rng = np.random.default_rng(42)
    v = rng.standard_normal(H.Ns)
    v /= np.linalg.norm(v)
    for _ in range(n_warmup):
        H.dot(v)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        out = H.dot(v)
    dt = (time.perf_counter() - t0) / n_calls
    _ = float(out.sum())
    return dt


def time_scipy_apply(H_csr, n_warmup: int = 1, n_calls: int = 5) -> float:
    rng = np.random.default_rng(42)
    v = rng.standard_normal(H_csr.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(n_warmup):
        H_csr @ v
    t0 = time.perf_counter()
    for _ in range(n_calls):
        out = H_csr @ v
    dt = (time.perf_counter() - t0) / n_calls
    _ = float(out.sum())
    return dt


def time_quspin_groundstate(H) -> tuple[float, float]:
    from scipy.sparse.linalg import eigsh
    A = H.tocsr()
    eigsh(A, k=1, which="SA", tol=1e-10, return_eigenvectors=False)  # warmup
    t0 = time.perf_counter()
    e0 = eigsh(A, k=1, which="SA", tol=1e-10, return_eigenvectors=False)[0]
    return time.perf_counter() - t0, float(e0)


# ============================================================================
# Distributed driver via ed_distributed_main
# ============================================================================

def time_distributed_lanczos(ed_dist_bin: str, mpiexec: str, np_ranks: int,
                             N: int, omp_threads: int,
                             gpu: bool = False) -> tuple[float, float]:
    """Invoke ed_distributed_main with --mode=lanczos (CPU or GPU) and parse its
    stdout for elapsed_s + eig[0]. Returns (seconds, e0).

    Setting gpu=True adds the --gpu flag, which routes through
    `distributed_lanczos_gpu` (cuBLAS local + NCCL collectives). Requires the
    binary to be built with WITH_CUDA=ON and NCCL available at runtime; if
    those preconditions are not met the binary will fail and we return
    (None, None)."""
    import re
    if not os.path.exists(ed_dist_bin) or shutil.which(mpiexec) is None:
        return (None, None)
    env = {**os.environ, "OMP_NUM_THREADS": str(omp_threads),
           "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"}
    cmd = [mpiexec, "-n", str(np_ranks), ed_dist_bin,
           "--mode", "lanczos", "--N", str(N), "--periodic", "1",
           "--max-iter", "200"]
    if gpu:
        cmd += ["--gpu", "--gpu-resident-spmv"]
    try:
        # warm up the cache once with a quick run; ignore the result.
        subprocess.run(cmd, capture_output=True, text=True, env=env,
                       timeout=120, check=False)
        # Take the median of three timed runs to absorb mpiexec startup jitter.
        wall_samples = []
        e0 = None
        for _ in range(3):
            t0 = time.perf_counter()
            result = subprocess.run(cmd, capture_output=True, text=True, env=env,
                                    timeout=600, check=True)
            wall_samples.append(time.perf_counter() - t0)
            # Prefer ed_distributed_main's own elapsed_s (excludes MPI startup).
            for line in result.stdout.splitlines():
                m = re.search(r"elapsed_s=([0-9.eE+-]+)", line)
                if m:
                    wall_samples[-1] = float(m.group(1))
                m2 = re.search(r"eig\[0\]=([-0-9.eE+]+)", line)
                if m2:
                    e0 = float(m2.group(1))
        wall_samples.sort()
        return (wall_samples[len(wall_samples) // 2], e0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return (None, None)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--build-dir", default="build",
                    help="Directory containing the bench binaries.")
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[12, 14, 16, 18])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--mpi-ranks", type=int, nargs="+", default=[1, 2, 4],
                    help="Distributed (CPU) sweep rank counts (skipped if mpiexec absent).")
    ap.add_argument("--gpu-mpi-ranks", type=int, nargs="+", default=[1, 2],
                    help="Distributed GPU+MPI sweep rank counts (one rank per visible GPU; "
                         "skipped if mpiexec absent or binary not built with WITH_CUDA + NCCL).")
    ap.add_argument("--mpiexec", default="mpiexec")
    ap.add_argument("--output", default="bench_all_backends.json")
    ap.add_argument("--skip-quspin", action="store_true")
    ap.add_argument("--skip-distributed", action="store_true")
    ap.add_argument("--skip-distributed-gpu", action="store_true",
                    help="Skip the GPU+MPI sweep even if --gpu-mpi-ranks is set.")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--min-time", default="0.4s")
    args = ap.parse_args()

    cpu_apply_bin = os.path.join(args.build_dir, "benchmarks", "bench_operator_apply")
    cpu_lan_bin   = os.path.join(args.build_dir, "benchmarks", "bench_lanczos_ground_state")
    gpu_apply_bin = os.path.join(args.build_dir, "benchmarks", "bench_gpu_operator_apply")
    gpu_lan_bin   = os.path.join(args.build_dir, "benchmarks", "bench_gpu_lanczos_ground_state")
    ed_dist_bin   = os.path.join(args.build_dir, "ed_distributed_main")

    sizes_alt = "|".join(str(n) for n in args.sizes)
    env_threads = {"OMP_NUM_THREADS": str(args.threads),
                   "OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"}

    # ---- C++ CPU benchmarks ------------------------------------------------
    print(f"[cpu] running CPU apply / lanczos benchmarks (threads={args.threads}, "
          f"min_time={args.min_time})...", flush=True)
    cpu_apply_cpx  = gbench_run(cpu_apply_bin,
                                f"^BM_OperatorApply_PBC/({sizes_alt})$",
                                env_threads, args.min_time)
    cpu_apply_real = gbench_run(cpu_apply_bin,
                                f"^BM_OperatorApply_PBC_Real/({sizes_alt})$",
                                env_threads, args.min_time)
    cpu_lanczos    = gbench_run(cpu_lan_bin,
                                f"^BM_LanczosGroundState/({sizes_alt})/",
                                env_threads, args.min_time)

    # ---- C++ GPU benchmarks ------------------------------------------------
    gpu_apply_pbc = []
    gpu_lanczos = []
    if not args.skip_gpu:
        print(f"[gpu] running GPU apply / lanczos benchmarks "
              f"(min_time={args.min_time})...", flush=True)
        gpu_apply_pbc = gbench_run(gpu_apply_bin,
                                   f"^BM_GPUOperatorApply_PBC/({sizes_alt})$",
                                   env_threads, args.min_time)
        gpu_lanczos   = gbench_run(gpu_lan_bin,
                                   f"^BM_GPULanczosGroundState/({sizes_alt})/",
                                   env_threads, args.min_time)

    # ---- QuSpin / scipy ----------------------------------------------------
    quspin_apply_us: dict[int, float | None] = {n: None for n in args.sizes}
    quspin_lanczos_ms: dict[int, float | None] = {n: None for n in args.sizes}
    quspin_e0: dict[int, float | None] = {n: None for n in args.sizes}
    scipy_apply_us: dict[int, float | None] = {n: None for n in args.sizes}
    if not args.skip_quspin:
        print("[quspin] running QuSpin apply + scipy.eigsh ground state...", flush=True)
        for N in args.sizes:
            try:
                H = quspin_chain(N)
                quspin_apply_us[N] = time_quspin_apply(H) * 1e6
                csr = H.tocsr()
                scipy_apply_us[N]  = time_scipy_apply(csr) * 1e6
                t_lan, e0 = time_quspin_groundstate(H)
                quspin_lanczos_ms[N] = t_lan * 1e3
                quspin_e0[N] = e0
            except Exception as exc:
                print(f"[quspin] N={N} failed: {exc}", flush=True)

    # ---- distributed (CPU) via mpiexec ------------------------------------
    distributed_results: dict[int, dict[int, dict[str, float | None]]] = {}
    if not args.skip_distributed:
        for np_ranks in args.mpi_ranks:
            distributed_results[np_ranks] = {}
            for N in args.sizes:
                t_dist, e0_dist = time_distributed_lanczos(
                    ed_dist_bin, args.mpiexec, np_ranks, N,
                    omp_threads=max(1, args.threads // np_ranks))
                distributed_results[np_ranks][N] = {
                    "wall_s": t_dist, "e0": e0_dist,
                }
                if t_dist is not None:
                    print(f"[mpi  np={np_ranks}] N={N} -> "
                          f"wall={t_dist:.3f}s, e0={e0_dist}", flush=True)
                else:
                    print(f"[mpi  np={np_ranks}] N={N} -> skipped", flush=True)

    # ---- distributed (GPU + NCCL) via mpiexec -----------------------------
    distributed_gpu_results: dict[int, dict[int, dict[str, float | None]]] = {}
    if not (args.skip_distributed or args.skip_distributed_gpu or args.skip_gpu):
        for np_ranks in args.gpu_mpi_ranks:
            distributed_gpu_results[np_ranks] = {}
            for N in args.sizes:
                t_dist, e0_dist = time_distributed_lanczos(
                    ed_dist_bin, args.mpiexec, np_ranks, N,
                    omp_threads=max(1, args.threads // np_ranks),
                    gpu=True)
                distributed_gpu_results[np_ranks][N] = {
                    "wall_s": t_dist, "e0": e0_dist,
                }
                if t_dist is not None:
                    print(f"[mpi+gpu np={np_ranks}] N={N} -> "
                          f"wall={t_dist:.3f}s, e0={e0_dist}", flush=True)
                else:
                    print(f"[mpi+gpu np={np_ranks}] N={N} -> skipped "
                          "(requires WITH_CUDA + NCCL + GPUs)", flush=True)

    # ---- assemble result rows ---------------------------------------------
    rows = []
    for N in args.sizes:
        dim = 1 << N
        rows.append({
            "N": N,
            "dim": dim,
            "cpu_apply_complex_us": gbench_time_for_size(cpu_apply_cpx,  N, "BM_OperatorApply_PBC"),
            "cpu_apply_real_us":    gbench_time_for_size(cpu_apply_real, N, "BM_OperatorApply_PBC_Real"),
            "cpu_lanczos_ms":       (gbench_time_for_lanczos(cpu_lanczos, N) or 0) / 1e3 or None,
            "gpu_apply_us":         gbench_time_for_size(gpu_apply_pbc,  N, "BM_GPUOperatorApply_PBC"),
            "gpu_lanczos_ms":       (gbench_time_for_lanczos(gpu_lanczos, N) or 0) / 1e3 or None,
            "quspin_apply_us":      quspin_apply_us[N],
            "scipy_apply_us":       scipy_apply_us[N],
            "quspin_lanczos_ms":    quspin_lanczos_ms[N],
            "e0_quspin":            quspin_e0[N],
            "distributed":          {str(np_): distributed_results[np_].get(N)
                                     for np_ in distributed_results} if distributed_results else {},
            "distributed_gpu":      {str(np_): distributed_gpu_results[np_].get(N)
                                     for np_ in distributed_gpu_results} if distributed_gpu_results else {},
        })

    # ---- write JSON --------------------------------------------------------
    payload = {
        "threads": args.threads,
        "sizes": args.sizes,
        "mpi_ranks": args.mpi_ranks,
        "gpu_mpi_ranks": args.gpu_mpi_ranks,
        "min_time": args.min_time,
        "rows": rows,
    }
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)

    # ---- pretty print ------------------------------------------------------
    print()
    print("=" * 110)
    print("Per-call SpMV (lower is better, microseconds). Heisenberg PBC chain, "
          f"threads = {args.threads}.")
    print("=" * 110)
    hdr = (f"{'N':>3} {'dim':>9} | "
           f"{'cpu cpx':>9} {'cpu real':>9} | "
           f"{'gpu':>9} | "
           f"{'QuSpin':>10} {'scipy':>10} | "
           f"{'speedup vs QS':>14} {'speedup vs scipy':>16}")
    print(hdr)
    print("-" * len(hdr))
    def f(x): return f"{x:.1f}" if x else "    -"
    for r in rows:
        cpu = r["cpu_apply_complex_us"]
        qs = r["quspin_apply_us"]
        sp = r["scipy_apply_us"]
        sp_qs = (qs / cpu) if (cpu and qs) else None
        sp_sp = (sp / cpu) if (cpu and sp) else None
        print(f"{r['N']:>3} {r['dim']:>9,} | "
              f"{f(cpu):>9} {f(r['cpu_apply_real_us']):>9} | "
              f"{f(r['gpu_apply_us']):>9} | "
              f"{f(qs):>10} {f(sp):>10} | "
              f"{(f'{sp_qs:.2f}x' if sp_qs else '   -'):>14} "
              f"{(f'{sp_sp:.2f}x' if sp_sp else '   -'):>16}")

    print()
    print("=" * 110)
    print("Ground-state Lanczos (lower is better, milliseconds, tol=1e-10).")
    print("=" * 110)
    hdr2 = (f"{'N':>3} {'dim':>9} | "
            f"{'cpu':>10} {'gpu':>10} {'QuSpin':>10} | "
            f"{'speedup vs QS':>14} | "
            f"{'E0 (QuSpin)':>14}")
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        cpu = r["cpu_lanczos_ms"]; gpu = r["gpu_lanczos_ms"]; qs = r["quspin_lanczos_ms"]
        sp = (qs / cpu) if (cpu and qs) else None
        e0 = r["e0_quspin"]
        print(f"{r['N']:>3} {r['dim']:>9,} | "
              f"{f(cpu):>10} {f(gpu):>10} {f(qs):>10} | "
              f"{(f'{sp:.2f}x' if sp else '   -'):>14} | "
              f"{f(e0):>14}")

    if distributed_results:
        print()
        print("=" * 110)
        print("Distributed Lanczos (CPU, ed_distributed_main via mpiexec, wall seconds).")
        print("=" * 110)
        hdr3 = f"{'N':>3} {'dim':>9} | " + "  ".join(
            [f"{'np='+str(np_):>12}" for np_ in args.mpi_ranks])
        print(hdr3)
        print("-" * len(hdr3))
        for r in rows:
            cells = []
            for np_ in args.mpi_ranks:
                d = r["distributed"].get(str(np_))
                if d and d.get("wall_s"):
                    cells.append(f"{d['wall_s']:.3f}")
                else:
                    cells.append("    -")
            print(f"{r['N']:>3} {r['dim']:>9,} | " + "  ".join(
                [f"{c:>12}" for c in cells]))

    if distributed_gpu_results:
        print()
        print("=" * 110)
        print("Distributed Lanczos (GPU + NCCL, ed_distributed_main --gpu via mpiexec, "
              "wall seconds).")
        print("=" * 110)
        hdr4 = f"{'N':>3} {'dim':>9} | " + "  ".join(
            [f"{'np='+str(np_):>12}" for np_ in args.gpu_mpi_ranks])
        print(hdr4)
        print("-" * len(hdr4))
        for r in rows:
            cells = []
            for np_ in args.gpu_mpi_ranks:
                d = r["distributed_gpu"].get(str(np_))
                if d and d.get("wall_s"):
                    cells.append(f"{d['wall_s']:.3f}")
                else:
                    cells.append("    -")
            print(f"{r['N']:>3} {r['dim']:>9,} | " + "  ".join(
                [f"{c:>12}" for c in cells]))


if __name__ == "__main__":
    main()
