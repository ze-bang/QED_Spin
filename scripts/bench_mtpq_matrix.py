"""bench_mtpq_matrix.py -- Phase-I follow-up benchmark (May 2026).

Times mTPQ on the same Heisenberg ring across the 4 x 2 matrix:

                       CPU         GPU
   no symmetry         x           x       (full 2^N Hilbert, one big call)
   fixed-Sz only       x           x       (iterate n_up = 0..N)
   symmetry only       x           x       (iterate Z_N irrep sectors)
   symmetry + Sz       x           x       (iterate irrep x n_up sub-sectors)

All cells compute the SAME physical quantity (Z-recombined whole-Hilbert
thermodynamics on the same Heisenberg ring), so total wall time is the
honest comparison metric: it answers "given Heisenberg ring N and a
target T grid, how long does mTPQ take in each (basis, backend) cell?".

Run with:
    python scripts/bench_mtpq_matrix.py --N 12 --max_iter 50
    python scripts/bench_mtpq_matrix.py --N 12 --max_iter 50 --skip_gpu

Phase F of the "Kill the GPU State-Lookup Hash" plan (May 2026):
the env var ``BENCH_HUGE=1`` adds a second pass at N=24, n_up=12,
exercising the dim ~2.7M regime where the legacy hash build itself
dominated and the rank-table rewrite delivers its biggest win. The
huge pass uses smaller mTPQ knobs (one sample, max_iter=10) so the
total bench stays under ~10 minutes even with the GPU+symmetry+Sz
cell running all 24 sectors. Disabled by default to keep the
default invocation cheap.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
import warnings
from typing import Any

import qed
from qed.symmetry import group_from_generators
from qed.workflow import _write_operator_directory, _write_symmetry_directory


def _heisenberg_ring(N: int):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    return b.to_operator()


def _ring_directory(N: int, with_symmetry: bool) -> str:
    H = _heisenberg_ring(N)
    tmp = tempfile.mkdtemp(prefix=f"bench_mtpq_N{N}_")
    _write_operator_directory(H, tmp)
    if with_symmetry:
        info = group_from_generators(
            N, [[(i + 1) % N for i in range(N)]]
        )
        _write_symmetry_directory(tmp, info)
    return tmp


def _run_one(*, label: str, N: int, device: str,
             with_sz: bool, with_symmetry: bool,
             max_iter: int, num_samples: int, num_T: int,
             T_min: float, T_max: float) -> dict[str, Any]:
    """Run one cell of the matrix and return wall-time + lane + E_gs.

    All cells call ``qed.thermal(method='mTPQ', ...)`` on the same
    Heisenberg ring. What varies between cells is whether we pass the
    in-memory operator (no symmetry) or a directory fixture (symmetry
    available), and the two ``use_*`` toggles.
    """
    out = {
        "label": label,
        "N": N,
        "device": device,
        "with_sz": with_sz,
        "with_symmetry": with_symmetry,
        "wall_seconds": float("nan"),
        "lane": "?",
        "E_gs": float("nan"),
        "samples": num_samples,
        "max_iter": max_iter,
        "ok": False,
        "error": "",
    }

    tmp_dir = None
    try:
        if with_symmetry:
            tmp_dir = _ring_directory(N, with_symmetry=True)
            H_arg: Any = tmp_dir
            kw_dir = dict(num_sites=N, spin=0.5)
        else:
            H_arg = _heisenberg_ring(N)
            kw_dir = {}

        # We pin to a small num_T / num_samples / max_iter to keep total
        # bench time bounded; mTPQ's hot loop scales linearly in all
        # three, so a small constant set keeps each cell honest.
        #
        # We capture RuntimeWarning's instead of silencing them: the
        # GPU promoter in the Pybind11 layer emits one of those when
        # an ``allow_gpu=True`` request silently fell back to CPU,
        # which is exactly the diagnostic we want for the bench.
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = qed.thermal(
                H_arg,
                method="mTPQ",
                T_min=T_min,
                T_max=T_max,
                num_T=num_T,
                num_samples=num_samples,
                max_iterations=max_iter,
                use_sz_if_conserved=with_sz,
                use_symmetry_if_available=with_symmetry,
                device=device,
                verbose=False,
                **kw_dir,
            )
        t1 = time.perf_counter()

        out["wall_seconds"] = t1 - t0
        # ThermalResult is a Python-facade aggregate over multiple
        # per-sector C++ EDResults and does not currently carry a
        # ``backend.lane``. We infer the effective lane from:
        #   1. the device the user asked for,
        #   2. any silent-fallback RuntimeWarning the GPU promoter
        #      emitted ("GPU requested but ran on CPU because ...").
        # Cell-level lane is a *summary* -- on a multi-sector run with
        # heterogeneous lanes we tag it "mixed".
        warned = [
            str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)
        ]
        gpu_fallbacks = [m for m in warned if "GPU" in m or "gpu" in m]
        if device == "gpu" and gpu_fallbacks:
            out["lane"] = "cpu (fallback)"
        else:
            out["lane"] = device
        out["warnings"] = len(warned)
        # ground-state energy isn't a primary mTPQ output (TPQ is a
        # finite-T sampler), but ``qed.thermal`` exposes it when the
        # downstream Z-recombination resolves it -- treat it as best-
        # effort sanity check.
        out["E_gs"] = float(getattr(res, "ground_state_energy", float("nan")))
        out["ok"] = True
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return out


def _run_matrix(*, N: int, max_iter: int, num_samples: int,
                num_T: int, T_min: float, T_max: float,
                skip_gpu: bool, label: str = "") -> list[dict[str, Any]]:
    """Run the full 4 x 2 basis-x-backend matrix at a single N. Pulled
    out of ``main`` so the Phase F BENCH_HUGE pass can re-invoke the
    same loop at N=24 without duplicating the table-printing code."""
    cells = [
        ("none",     False, False),
        ("sz",       True,  False),
        ("sym",      False, True),
        ("sym+sz",   True,  True),
    ]
    devices = ["cpu"] + ([] if skip_gpu else ["gpu"])

    tag = f" [{label}]" if label else ""
    print(f"# bench_mtpq_matrix{tag}")
    print(f"# N={N}  T=[{T_min},{T_max}]  num_T={num_T}  "
          f"num_samples={num_samples}  max_iter={max_iter}")
    print(f"# qed.has_cuda_build()={qed.has_cuda_build()}")
    print(f"# ED_GPU_USE_HASH={os.environ.get('ED_GPU_USE_HASH', '0')}  "
          f"ED_GPU_STORE_BASIS={os.environ.get('ED_GPU_STORE_BASIS', '1')}")
    print()
    header = ("basis", "device", "lane", "wall_s",
              "E_gs", "samples", "max_iter", "status")
    print("\t".join(header))
    sys.stdout.flush()

    results = []
    for cell_label, with_sz, with_sym in cells:
        for device in devices:
            r = _run_one(
                label=cell_label, N=N, device=device,
                with_sz=with_sz, with_symmetry=with_sym,
                max_iter=max_iter,
                num_samples=num_samples,
                num_T=num_T,
                T_min=T_min,
                T_max=T_max,
            )
            results.append(r)
            status = "ok" if r["ok"] else "ERR: " + r["error"][:80]
            print(
                f"{cell_label}\t{device}\t{r['lane']}\t"
                f"{r['wall_seconds']:.3f}\t"
                f"{r['E_gs']:.6f}\t"
                f"{r['samples']}\t{r['max_iter']}\t{status}"
            )
            sys.stdout.flush()

    print()
    print("# CPU vs GPU per basis (wall_cpu / wall_gpu)")
    print("basis\tcpu_s\tgpu_s\tspeedup\tcpu_lane\tgpu_lane")
    by = {(r["label"], r["device"]): r for r in results}
    for cell_label, _, _ in cells:
        c = by.get((cell_label, "cpu"))
        g = by.get((cell_label, "gpu"))
        if c is None or g is None:
            continue
        if not (c["ok"] and g["ok"]):
            cpu_s = f"{c['wall_seconds']:.3f}" if c["ok"] else "ERR"
            gpu_s = f"{g['wall_seconds']:.3f}" if g["ok"] else "ERR"
            print(f"{cell_label}\t{cpu_s}\t{gpu_s}\t-\t"
                  f"{c['lane']}\t{g['lane']}")
            continue
        speedup = c["wall_seconds"] / max(g["wall_seconds"], 1e-9)
        print(f"{cell_label}\t{c['wall_seconds']:.3f}\t"
              f"{g['wall_seconds']:.3f}\t"
              f"{speedup:.2f}x\t{c['lane']}\t{g['lane']}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=12)
    p.add_argument("--max_iter", type=int, default=50,
                   help="mTPQ max iterations (default 50)")
    p.add_argument("--num_samples", type=int, default=2,
                   help="mTPQ number of random samples (default 2)")
    p.add_argument("--num_T", type=int, default=4,
                   help="Number of T grid points (default 4)")
    p.add_argument("--T_min", type=float, default=0.5)
    p.add_argument("--T_max", type=float, default=4.0)
    p.add_argument("--skip_gpu", action="store_true")
    args = p.parse_args()

    _run_matrix(
        N=args.N,
        max_iter=args.max_iter,
        num_samples=args.num_samples,
        num_T=args.num_T,
        T_min=args.T_min,
        T_max=args.T_max,
        skip_gpu=args.skip_gpu,
        label=f"N={args.N}",
    )

    # Phase F of the "Kill the GPU State-Lookup Hash" plan (May 2026):
    # opt-in heavy run at N=24, n_up=12 (dim_full_sz = C(24, 12) = 2.7M)
    # where the legacy hash build itself dominates wall time. The
    # smaller knobs (1 sample, max_iter=10) keep this pass bounded
    # while still demonstrating the post-rank-patch perf win.
    huge_raw = os.environ.get("BENCH_HUGE", "").strip()
    if huge_raw and huge_raw != "0":
        print()
        print("# " + "=" * 70)
        print("# Phase F follow-up: BENCH_HUGE=1 at N=24, n_up=12 "
              "(dim_full_sz=2.7M)")
        print("# " + "=" * 70)
        _run_matrix(
            N=24,
            max_iter=10,
            num_samples=1,
            num_T=args.num_T,
            T_min=args.T_min,
            T_max=args.T_max,
            skip_gpu=args.skip_gpu,
            label="huge N=24",
        )


if __name__ == "__main__":
    main()
