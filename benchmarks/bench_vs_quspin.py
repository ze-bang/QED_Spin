#!/usr/bin/env python3
"""
benchmarks/bench_vs_quspin.py

Head-to-head SOTA-peer benchmark for the matrix-free Heisenberg-chain SpMV
and the ground-state Lanczos.

Peers:
  - **us (apply)**     : Operator::apply complex path  (this repo, post-audit).
  - **us (apply_real)**: Operator::apply_real real path (audit §2.1 Phase 1).
  - **QuSpin**         : weisses-standard Python+C++ ED library.
  - **scipy.sparse**   : CSR + ARPACK (the universal reference everyone cites).

For each system size N (1D periodic Heisenberg ring) we time:
  * one matrix-vector product (dim x dim acting on a normalized real vector),
  * a full ground-state Lanczos to tol=1e-10 (eigenvalues only).

The C++ side is read from `bench_operator_apply` and `bench_lanczos_ground_state`
JSON output produced by Google Benchmark; this script invokes them with
`--benchmark_format=json` and parses the result.

Usage:
  python benchmarks/bench_vs_quspin.py            \\
      --build-dir build/benchmarks                \\
      --sizes 12 14 16 18                          \\
      --threads $(nproc)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

import numpy as np


# --------------------------- helpers ---------------------------------------

def run_cpp_bench(binary: str, filter_re: str, env: dict[str, str]) -> list[dict[str, Any]]:
    """Run a Google-Benchmark binary, capture JSON, return its 'benchmarks' list.

    Note: lanczos() prints progress to stdout, so we route the JSON to a file
    via --benchmark_out= and let stdout chatter go to /dev/null.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        out_path = f.name
    cmd = [
        binary,
        f"--benchmark_out={out_path}",
        "--benchmark_out_format=json",
        f"--benchmark_filter={filter_re}",
        "--benchmark_min_time=0.4s",
    ]
    subprocess.run(cmd, env={**os.environ, **env},
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   check=True)
    with open(out_path) as fh:
        j = json.load(fh)
    os.unlink(out_path)
    return j["benchmarks"]


def quspin_heisenberg_pbc(N: int):
    """1D Heisenberg chain with PBC, spin-1/2, in QuSpin (no symmetries)."""
    from quspin.basis import spin_basis_general
    from quspin.operators import hamiltonian

    basis = spin_basis_general(N=N, S="1/2", pauli=False)
    Jzz = [[1.0, i, (i + 1) % N] for i in range(N)]
    Jpm = [[0.5, i, (i + 1) % N] for i in range(N)]
    static = [["zz", Jzz], ["+-", Jpm], ["-+", Jpm]]
    H = hamiltonian(static, [], basis=basis,
                    dtype=np.float64, check_symm=False, check_herm=False, check_pcon=False)
    return H, basis


def time_quspin_apply(H, n_calls: int = 5) -> float:
    dim = H.Ns
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    H.dot(v)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_calls):
        out = H.dot(v)
    dt = (time.perf_counter() - t0) / n_calls
    _ = out.sum()
    return dt


def time_quspin_groundstate(H, n_calls: int = 1) -> tuple[float, float]:
    """Time eigsh(k=1, which='SA'); return (seconds_per_run, ground_state_energy)."""
    from scipy.sparse.linalg import eigsh
    A = H.tocsr()
    e0_check = eigsh(A, k=1, which="SA", tol=1e-10, return_eigenvectors=False)[0]
    t0 = time.perf_counter()
    for _ in range(n_calls):
        e0 = eigsh(A, k=1, which="SA", tol=1e-10, return_eigenvectors=False)[0]
    dt = (time.perf_counter() - t0) / n_calls
    return dt, float(e0)


def time_scipy_apply(H_csr, n_calls: int = 5) -> float:
    dim = H_csr.shape[0]
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    H_csr @ v  # warmup
    t0 = time.perf_counter()
    for _ in range(n_calls):
        out = H_csr @ v
    dt = (time.perf_counter() - t0) / n_calls
    _ = out.sum()
    return dt


# ----------------------------- main ----------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build/benchmarks",
                    help="Directory containing bench_operator_apply / "
                         "bench_lanczos_ground_state binaries.")
    ap.add_argument("--sizes", type=int, nargs="+", default=[12, 14, 16, 18])
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--skip-cpp", action="store_true",
                    help="Skip the C++ bench step (use cached JSON if present).")
    ap.add_argument("--cache-json", default="bench_cpp_results.json")
    args = ap.parse_args()

    env = {"OMP_NUM_THREADS": str(args.threads)}

    apply_bin = os.path.join(args.build_dir, "bench_operator_apply")
    lancz_bin = os.path.join(args.build_dir, "bench_lanczos_ground_state")

    cpp = {}
    if not args.skip_cpp:
        # Google Benchmark filter accepts a single regex; we match any of the
        # requested sizes via "(8|10|...)$" -- $ keeps "16" from also matching
        # "160". For Lanczos the suffix is "/N/krylov" so we don't anchor.
        sizes_alt = "|".join(str(n) for n in args.sizes)
        cpp_apply_pbc      = run_cpp_bench(apply_bin, f"^BM_OperatorApply_PBC/({sizes_alt})$",      env)
        cpp_apply_pbc_real = run_cpp_bench(apply_bin, f"^BM_OperatorApply_PBC_Real/({sizes_alt})$", env)
        cpp_apply_real     = run_cpp_bench(apply_bin, f"^BM_OperatorApplyReal_PBC/({sizes_alt})$",  env)
        cpp_lanczos        = run_cpp_bench(lancz_bin, f"^BM_LanczosGroundState/({sizes_alt})/",     env)
        cpp = {
            "apply_pbc": cpp_apply_pbc,
            "apply_pbc_real": cpp_apply_pbc_real,
            "apply_real_direct": cpp_apply_real,
            "lanczos": cpp_lanczos,
        }
        with open(args.cache_json, "w") as f:
            json.dump(cpp, f, indent=2)
    else:
        with open(args.cache_json) as f:
            cpp = json.load(f)

    # ---- pick out per-N results from C++ JSON ---------------------------
    def cpp_time_us(bench_list: list[dict[str, Any]], N: int) -> float | None:
        suffix = f"/{N}"
        for b in bench_list:
            if b["name"].endswith(suffix):
                return float(b["real_time"])  # microseconds (kMicrosecond unit)
        return None

    def cpp_lanczos_ms(bench_list: list[dict[str, Any]], N: int) -> float | None:
        # name format: BM_LanczosGroundState/{N}/{krylov}[/min_time:...]
        # We always pull the second '/'-separated part (index 1).
        for b in bench_list:
            parts = b["name"].split("/")
            if len(parts) >= 2 and parts[1] == str(N):
                return float(b["real_time"])
        return None

    rows = []
    for N in args.sizes:
        dim = 1 << N
        print(f"\n=== N = {N}  (dim = {dim:,}) ===", flush=True)

        # ---- QuSpin
        try:
            H_qs, _ = quspin_heisenberg_pbc(N)
            t_qs_apply = time_quspin_apply(H_qs, n_calls=max(3, 50 // (1 + dim // 1024)))
            t_qs_lan, e0_qs = time_quspin_groundstate(H_qs)
        except Exception as e:
            print(f"  QuSpin failed: {e}")
            t_qs_apply = None
            t_qs_lan = None
            e0_qs = None
            H_qs = None

        # ---- scipy.sparse from QuSpin's CSR (fallback if QuSpin failed)
        if H_qs is not None:
            H_csr = H_qs.tocsr()
            t_sp_apply = time_scipy_apply(H_csr, n_calls=max(3, 50 // (1 + dim // 1024)))
        else:
            t_sp_apply = None

        # ---- our C++ numbers
        t_us_apply        = cpp_time_us(cpp.get("apply_pbc", []), N)        # us
        t_us_apply_real   = cpp_time_us(cpp.get("apply_pbc_real", []), N)
        t_us_apply_direct = cpp_time_us(cpp.get("apply_real_direct", []), N)
        t_us_lan_ms       = cpp_lanczos_ms(cpp.get("lanczos", []), N)

        rows.append({
            "N": N, "dim": dim,
            "us_apply_us":        t_us_apply,
            "us_apply_real_us":   t_us_apply_real,
            "us_apply_direct_us": t_us_apply_direct,
            "quspin_apply_us":    t_qs_apply * 1e6 if t_qs_apply is not None else None,
            "scipy_apply_us":     t_sp_apply * 1e6 if t_sp_apply is not None else None,
            "us_lanczos_ms":      t_us_lan_ms,
            "quspin_lanczos_ms":  t_qs_lan * 1e3 if t_qs_lan is not None else None,
            "e0_quspin":          e0_qs,
        })

    # ---- pretty print --------------------------------------------------
    print()
    print("=" * 96)
    print("Per-call SpMV time (lower is better). Heisenberg PBC chain, threads = "
          f"{args.threads}.")
    print("=" * 96)
    hdr = f"{'N':>3} {'dim':>10} | {'us apply (cpx) us':>17} | {'us apply (real) us':>18} | {'us apply_real us':>16} | {'QuSpin us':>10} | {'scipy.sp us':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(x):
            return f"{x:.1f}" if x is not None else "    -"
        print(f"{r['N']:>3} {r['dim']:>10,} | {f(r['us_apply_us']):>17} | "
              f"{f(r['us_apply_real_us']):>18} | {f(r['us_apply_direct_us']):>16} | "
              f"{f(r['quspin_apply_us']):>10} | {f(r['scipy_apply_us']):>11}")

    print()
    print("=" * 96)
    print(f"Ground-state Lanczos (tol=1e-10).  ms per full Lanczos call.")
    print("=" * 96)
    hdr = f"{'N':>3} {'dim':>10} | {'us (ms)':>10} | {'QuSpin (ms)':>12} | {'speedup vs QuSpin':>18} | {'E0 (QuSpin)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(x, fmt="{:.2f}"):
            return fmt.format(x) if x is not None else "   -"
        sp = (r["quspin_lanczos_ms"] / r["us_lanczos_ms"]
              if (r["us_lanczos_ms"] and r["quspin_lanczos_ms"]) else None)
        print(f"{r['N']:>3} {r['dim']:>10,} | {f(r['us_lanczos_ms']):>10} | "
              f"{f(r['quspin_lanczos_ms']):>12} | "
              f"{(f'{sp:.2f}x' if sp is not None else '   -'):>18} | "
              f"{f(r['e0_quspin']):>14}")

    out_path = "bench_vs_quspin_results.json"
    with open(out_path, "w") as f:
        json.dump({"threads": args.threads, "rows": rows}, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
