"""Focused GPU sweep at larger N for the (workflow, sym) cells where
GPU has a working kernel and could actually beat CPU.

Skips:
    * Symm / Sz+Symm cells (CPU-fallback today; Phase 1c)
    * FT-LTLM / FT-KPM_DOS (no GPU kernel per `_SOLVER_DEVICE_KERNELS`)
    * DSSF-* + Symm (streaming-symmetry binding, CPU-only)

Runs:
    * GS, none/Sz
    * FT-FTLM, none/Sz
    * DSSF-GS, none/Sz
    * DSSF-FT, none/Sz

Default N=14 (Hilbert dim 16384) gives the GPU lane enough work to
amortise the Python subprocess overhead.

Usage:
    python benchmarks/bench_gpu_symmetry_focused.py --sizes 14
    python benchmarks/bench_gpu_symmetry_focused.py --sizes 14,16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Import the shared infra from the matrix benchmark.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_gpu_symmetry_matrix as M  # type: ignore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", type=str, default="14")
    p.add_argument(
        "--out", type=str,
        default="benchmarks/bench_gpu_symmetry_focused_results.json",
    )
    args = p.parse_args()

    if not M.cuda_available():
        print("FATAL: no CUDA build/device. Run on a CUDA-enabled host.",
              file=sys.stderr)
        sys.exit(2)

    sizes = [int(s) for s in args.sizes.split(",")]
    # Restrict to the (none, Sz) lanes which have a real GPU kernel
    # today. Symm/Sz+Symm fall through the streaming-symmetry binding
    # which has no GPU dispatch yet (Phase 1c); we already document
    # that gap in bench_gpu_symmetry_matrix.
    syms_with_gpu_kernel = ("none", "Sz")
    all_rows = []
    for N in sizes:
        group_size = N
        print(f"\n=== N={N} ===", flush=True)
        for sym in syms_with_gpu_kernel:
            for kind, label, extra in [
                ("gs", "GS", None),
                ("thermal", "FT-FTLM", {"method": "FTLM"}),
                ("dssf_gs", "DSSF-GS", {
                    "omega": [-2 + 8.0*i/29 for i in range(30)],
                    "eta": 0.15, "q_int": 1}),
                ("dssf_ft", "DSSF-FT", {
                    "omega": [-2 + 8.0*i/24 for i in range(25)],
                    "eta": 0.2, "T": [1.0], "q_int": 1}),
            ]:
                print(f"  {label} {sym}...", flush=True)
                cpu_out, gpu_out = M._cell(
                    kind, sym, N, extra=extra,
                    cpu_timeout=900.0, gpu_timeout=900.0,
                )
                all_rows.append(M._to_row(
                    label, sym, N, M._expected_dim(sym, N, group_size),
                    cpu_out, gpu_out,
                ))
    print()
    M._print_table(all_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
