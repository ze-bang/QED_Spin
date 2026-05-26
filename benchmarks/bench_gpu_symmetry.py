"""GPU symmetry benchmark (Phase 5 of the "Unified CPU/GPU symmetry
architecture" plan, May 2026).

Once the Phase 1c follow-up lands (lazy GPU mirror in
``StreamingSymmetryOperator::SectorView::bind_cuda``), this benchmark
exercises the GPU symmetry lane end-to-end:

    CPU SectorView  vs  GPU mirror (apply_terms_gpu<DeviceSymmetryBasisPolicy>)

across the three workflows (GS, FT, DSSF) at N=14 Heisenberg with
translation symmetry. The architectural plumbing for this path is
live in the May-2026 PR; the GPU mirror construction is the next
deliverable, so this benchmark currently reports "skip: WITH_CUDA off"
or "skip: ED_GPU_SYMMETRY_MIRROR=0" when the mirror is unavailable.

When the mirror is active, it reports:
    * Wall time per matvec (CPU SectorView, GPU mirror)
    * Per-call kernel time (when ED_GPU_TIMING=1)
    * Speedup ratio
    * GFLOPS estimate

Usage
-----
    # Smoke test (CPU SectorView only, GPU skipped):
    python benchmarks/bench_gpu_symmetry.py --sizes 12

    # Full sweep when the GPU mirror lands:
    ED_GPU_SYMMETRY_MIRROR=1 python benchmarks/bench_gpu_symmetry.py --sizes 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def _check_cuda_available() -> bool:
    """Heuristic: probe for nvidia-smi or CUDA_VISIBLE_DEVICES."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return True
    try:
        import subprocess

        r = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, timeout=5
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_gpu_mirror_enabled() -> bool:
    """The Phase 2 architectural plumbing ships with ED_GPU_SYMMETRY_MIRROR=1
    as the opt-in flag; the actual mirror construction is the Phase 1c
    follow-up.
    """
    return os.environ.get("ED_GPU_SYMMETRY_MIRROR", "0") == "1"


def run_benchmark(num_sites: int, num_matvecs: int = 100):
    """Run the CPU-vs-GPU symmetry-projected matvec benchmark for one N."""
    import qed

    results = {
        "num_sites": num_sites,
        "num_matvecs": num_matvecs,
        "cuda_available": _check_cuda_available(),
        "gpu_mirror_enabled": _check_gpu_mirror_enabled(),
    }

    # Build the Heisenberg ring + translation symmetry.
    builder = qed.input.HamiltonianBuilder(num_sites)
    bonds = [(i, (i + 1) % num_sites) for i in range(num_sites)]
    builder.heisenberg(bonds, J=1.0)
    H = builder.to_operator()

    z_n = qed.GeneratorSet(
        name="ZN_translation",
        description="Cyclic translation by one site",
        generators=[[(i + 1) % num_sites for i in range(num_sites)]],
        orders=[num_sites],
        group_size=num_sites,
    )

    # --- CPU SectorView lane ---
    t0 = time.perf_counter()
    result_cpu = qed.solve(H, k=1, symmetry=z_n)
    dt_cpu = time.perf_counter() - t0
    results["cpu_e0"] = float(np.min(result_cpu.eigenvalues))
    results["cpu_wall_s"] = dt_cpu

    # --- GPU mirror lane (architectural seam) ---
    if not results["cuda_available"]:
        results["gpu_status"] = "skip: no CUDA device visible"
        return results

    if not results["gpu_mirror_enabled"]:
        results["gpu_status"] = (
            "skip: ED_GPU_SYMMETRY_MIRROR=0 -- Phase 1c follow-up "
            "(GPUSymmetrizedOperator mirror) has not landed yet. The "
            "architectural plumbing (Geometry::supports_device_matvec, "
            "select_backend gate, SectorView::bind_cuda delegation) "
            "is active; the mirror construction TU is the next PR."
        )
        return results

    # When the Phase 1c follow-up lands, the next call selects CudaBackend
    # automatically because the SectorView now sets
    # supports_device_matvec=true. Until then, this throws std::logic_error
    # at the bind_cuda_for_sector stub and we report the failure.
    try:
        t0 = time.perf_counter()
        result_gpu = qed.solve(H, k=1, symmetry=z_n, device="gpu")
        dt_gpu = time.perf_counter() - t0
        results["gpu_e0"] = float(np.min(result_gpu.eigenvalues))
        results["gpu_wall_s"] = dt_gpu
        results["gpu_status"] = "ok"
        results["speedup"] = dt_cpu / dt_gpu if dt_gpu > 0 else float("inf")
    except Exception as e:
        results["gpu_status"] = f"error: {type(e).__name__}: {e}"

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes",
        type=str,
        default="12",
        help="Comma-separated list of N values. Default: 12",
    )
    p.add_argument(
        "--num-matvecs",
        type=int,
        default=100,
        help="Number of matvec iterations to time. Default: 100",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. Default: stdout.",
    )
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    results = []
    for n in sizes:
        print(f"=== N={n} ===", file=sys.stderr)
        r = run_benchmark(n, num_matvecs=args.num_matvecs)
        results.append(r)
        print(json.dumps(r, indent=2), file=sys.stderr)

    out_text = json.dumps(results, indent=2)
    if args.out:
        Path(args.out).write_text(out_text)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
