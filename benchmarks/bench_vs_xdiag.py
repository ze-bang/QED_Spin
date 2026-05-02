#!/usr/bin/env python3
"""benchmarks/bench_vs_xdiag.py

Head-to-head benchmark between this codebase and `XDiag.jl`
(https://github.com/awietek/XDiag.jl).

Both libraries solve the same canonical problem -- the 1D periodic
Heisenberg chain at spin 1/2, no symmetries -- and report:

  * One matrix-vector product `H @ v` on a unit-norm random vector (us).
  * The full ground-state Lanczos to numerical convergence (ms).

The XDiag side runs as a Julia subprocess (`bench_vs_xdiag.jl`) and
writes its results to a JSON file we splice in here. The qed
side uses `qed.lanczos` and a direct timing of `Operator.apply`.

Usage:

  # One-time: install XDiag.jl into a dedicated env (one-off, ~5min).
  julia --project=benchmarks/xdiag_env -e \
        'using Pkg; Pkg.add("XDiag"); Pkg.add("JSON")'

  # Run the head-to-head:
  python3 benchmarks/bench_vs_xdiag.py \
        --sizes 12 14 16 18 \
        --threads $(nproc) \
        --output bench_vs_xdiag.json

The optional ``--fixed-sz`` flag runs both peers in the
``Spinhalf(N, N/2)`` / ``FixedSzOperator(N, N//2)`` Sz=0 sector --
the natural "best case for an ED library that knows about
conservation laws" comparison.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================================
# qed peer
# ============================================================================


def _build_chain_op(N: int, fixed_sz: bool):
    """Return a ``qed`` (Fixed)SzOperator for the same Heisenberg PBC
    chain XDiag is given. We use the input library so the term layout matches
    XDiag's ``SdotS`` shortcut bit-for-bit (S+S- + S-S+ + SzSz, J=1)."""
    import qed as qed

    nn = [(i, (i + 1) % N) for i in range(N)]
    builder = qed.input.HamiltonianBuilder(num_sites=N)
    builder.heisenberg(bonds=nn, J=1.0)
    if fixed_sz:
        op = qed.FixedSzOperator(num_sites=N, n_up=N // 2, spin=0.5)
        builder.emit_into(op)
    else:
        op = builder.to_operator()
    return op


def time_qed_apply(N: int, fixed_sz: bool,
                          n_warmup: int = 1, n_calls: int = 5) -> float | None:
    """Median microseconds for one ``Operator.apply(v)`` on a unit-norm vector."""
    import qed as qed

    op = _build_chain_op(N, fixed_sz=fixed_sz)
    dim = op.dimension if hasattr(op, "dimension") else (1 << N)
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.complex128)
    v /= np.linalg.norm(v)
    apply = getattr(op, "apply", None)
    if apply is None:
        return None
    for _ in range(n_warmup):
        apply(v)
    samples_us: list[float] = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        out = apply(v)
        samples_us.append((time.perf_counter() - t0) * 1e6)
        _ = out.sum()  # keep result alive
    samples_us.sort()
    return statistics.median(samples_us)


def time_qed_lanczos(N: int, fixed_sz: bool,
                            n_calls: int = 3) -> tuple[float | None,
                                                       float | None]:
    """Median ground-state Lanczos time (to ``tol=1e-10``). Returns
    ``(milliseconds, eigenvalue)``.

    Mirrors the ``time_eigval0`` median-of-``n_calls`` approach used on
    the XDiag side: a single timed call is too noisy at large N
    (variance dominated by cache state and the OS scheduler).
    """
    import qed as qed

    op = _build_chain_op(N, fixed_sz=fixed_sz)
    # Warm up: triggers any first-touch allocation, JIT, etc.
    _ = qed.lanczos(op, max_iter=200, exct=1, tolerance=1e-10)
    samples_ms: list[float] = []
    e0: float | None = None
    for _ in range(n_calls):
        t0 = time.perf_counter()
        eigs = qed.lanczos(op, max_iter=400, exct=1, tolerance=1e-10)
        samples_ms.append((time.perf_counter() - t0) * 1e3)
        if hasattr(eigs, "eigenvalues"):
            eigs = eigs.eigenvalues
        if len(eigs) > 0:
            e0 = float(min(eigs))
    samples_ms.sort()
    dt_ms = samples_ms[len(samples_ms) // 2]
    return dt_ms, e0


# ============================================================================
# XDiag peer (driven via Julia subprocess)
# ============================================================================


def _resolve_julia(julia_binary: str | None) -> str:
    candidate = julia_binary or shutil.which("julia")
    if candidate is None:
        raise FileNotFoundError(
            "julia is not on $PATH. Install Julia (https://julialang.org/) and "
            "rerun, or pass --julia /abs/path/to/julia."
        )
    return candidate


def run_xdiag(sizes: list[int], *, julia: str, project_dir: Path,
              script: Path, threads: int, fixed_sz: bool,
              n_apply: int, output_json: Path) -> dict[str, Any]:
    """Invoke the Julia bench script. Returns the parsed JSON payload."""
    cmd = [
        julia, f"--project={project_dir}",
        f"--threads={threads}",
        str(script),
        "--sizes", ",".join(str(n) for n in sizes),
        "--threads", str(threads),
        "--n-apply", str(n_apply),
        "--output", str(output_json),
    ]
    if fixed_sz:
        cmd.append("--fixed-sz")
    print(f"[xdiag] $ {' '.join(cmd)}", flush=True)
    env = {**os.environ,
           "JULIA_NUM_THREADS": str(threads),
           "OMP_NUM_THREADS":   str(threads)}
    subprocess.run(cmd, env=env, check=True)
    with open(output_json) as f:
        return json.load(f)


# ============================================================================
# Pretty printing
# ============================================================================


def _fmt_us(x: float | None) -> str:
    if x is None:
        return "    -"
    if x >= 1000:
        return f"{x/1000:.2f} ms"
    return f"{x:.1f} us"


def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "    -"
    if x >= 1000:
        return f"{x/1000:.2f} s"
    return f"{x:.1f} ms"


def _speedup(us_t: float | None, peer_t: float | None) -> str:
    if us_t is None or peer_t is None or us_t == 0:
        return "    -"
    return f"{peer_t / us_t:.2f}x"


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 14, 16, 18])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--fixed-sz", action="store_true",
                    help="Run both peers in the Sz=0 sector (Spinhalf(N, N/2) "
                         "/ FixedSzOperator(N, N//2)).")
    ap.add_argument("--n-apply", type=int, default=5,
                    help="Number of timed apply() calls per size (median is "
                         "reported).")
    ap.add_argument("--julia", default=None,
                    help="Path to julia binary (default: $PATH lookup).")
    ap.add_argument("--julia-project", default="benchmarks/xdiag_env",
                    help="Julia env directory holding XDiag.jl + JSON.jl.")
    ap.add_argument("--julia-script", default="benchmarks/bench_vs_xdiag.jl")
    ap.add_argument("--xdiag-output", default="bench_vs_xdiag_xdiag.json")
    ap.add_argument("--output", default="bench_vs_xdiag.json")
    ap.add_argument("--skip-quantum-ed", action="store_true",
                    help="Only run the Julia/XDiag side (debug aid).")
    ap.add_argument("--skip-xdiag", action="store_true",
                    help="Only run the qed side (debug aid).")
    args = ap.parse_args()

    # ---- XDiag ---------------------------------------------------------
    xdiag_payload: dict[str, Any] = {"rows": []}
    if not args.skip_xdiag:
        julia = _resolve_julia(args.julia)
        project = Path(args.julia_project).resolve()
        script  = Path(args.julia_script).resolve()
        out_json = Path(args.xdiag_output).resolve()
        xdiag_payload = run_xdiag(
            args.sizes, julia=julia, project_dir=project,
            script=script, threads=args.threads,
            fixed_sz=args.fixed_sz, n_apply=args.n_apply,
            output_json=out_json,
        )

    xdiag_by_N = {row["N"]: row for row in xdiag_payload.get("rows", [])}

    # ---- qed ----------------------------------------------------
    rows = []
    for N in args.sizes:
        dim = (math.comb(N, N // 2) if args.fixed_sz else (1 << N))
        print(f"\n=== N = {N}  (dim = {dim:,})  fixed_sz={args.fixed_sz} ===",
              flush=True)
        if args.skip_qed:
            us_apply = None
            us_lanczos = None
            e0_us = None
        else:
            try:
                us_apply = time_qed_apply(N, args.fixed_sz,
                                                 n_calls=args.n_apply)
            except Exception as exc:
                print(f"  [qed apply] N={N} failed: {exc}")
                us_apply = None
            try:
                us_lanczos, e0_us = time_qed_lanczos(N, args.fixed_sz)
            except Exception as exc:
                print(f"  [qed lanczos] N={N} failed: {exc}")
                us_lanczos = None
                e0_us = None

        xd = xdiag_by_N.get(N, {})
        rows.append({
            "N": N,
            "dim": dim,
            "fixed_sz": args.fixed_sz,
            "us_apply_us":      us_apply,
            "us_lanczos_ms":    us_lanczos,
            "us_e0":            e0_us,
            "xdiag_apply_us":   xd.get("apply_us"),
            "xdiag_lanczos_ms": xd.get("lanczos_ms"),
            "xdiag_e0":         xd.get("e0"),
            "xdiag_ok":         bool(xd.get("ok", False)),
            "xdiag_error":      xd.get("error"),
        })
        # Quick correctness check.
        if e0_us is not None and xd.get("e0") is not None:
            d = abs(float(e0_us) - float(xd["e0"]))
            tag = "OK" if d < 1e-6 else "MISMATCH"
            print(f"  E0(qed) = {e0_us:.10f}, "
                  f"E0(xdiag) = {xd['e0']:.10f}, |delta| = {d:.2e}  [{tag}]",
                  flush=True)

    # ---- pretty print --------------------------------------------------
    print()
    print("=" * 96)
    print(f"Per-call SpMV (lower is better). 1D Heisenberg PBC chain, "
          f"threads = {args.threads}, fixed_sz = {args.fixed_sz}.")
    print("=" * 96)
    hdr = (f"{'N':>3} {'dim':>10} | {'qed':>12} | {'xdiag':>12} | "
           f"{'speedup':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['N']:>3} {r['dim']:>10,} | "
              f"{_fmt_us(r['us_apply_us']):>12} | "
              f"{_fmt_us(r['xdiag_apply_us']):>12} | "
              f"{_speedup(r['us_apply_us'], r['xdiag_apply_us']):>9}")

    print()
    print("=" * 96)
    print(f"Ground-state Lanczos. qed.lanczos vs XDiag.eigval0 (both at "
          f"natural numerical convergence).")
    print("=" * 96)
    hdr = (f"{'N':>3} {'dim':>10} | {'qed':>12} | {'xdiag':>12} | "
           f"{'speedup':>9} | {'E0 (xdiag)':>14}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['N']:>3} {r['dim']:>10,} | "
              f"{_fmt_ms(r['us_lanczos_ms']):>12} | "
              f"{_fmt_ms(r['xdiag_lanczos_ms']):>12} | "
              f"{_speedup(r['us_lanczos_ms'], r['xdiag_lanczos_ms']):>9} | "
              f"{r['xdiag_e0'] if r['xdiag_e0'] is not None else '-':>14}")

    payload = {
        "threads":   args.threads,
        "fixed_sz":  args.fixed_sz,
        "n_apply":   args.n_apply,
        "sizes":     args.sizes,
        "xdiag":     xdiag_payload,
        "rows":      rows,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
