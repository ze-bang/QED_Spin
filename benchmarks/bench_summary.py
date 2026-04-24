#!/usr/bin/env python3
"""
benchmarks/bench_summary.py

Generates a single-page Markdown summary of the head-to-head benchmark
between this codebase and QuSpin / scipy.sparse, drawing from
`bench_vs_quspin_results.json` (produced by bench_vs_quspin.py).

Usage:
    python benchmarks/bench_vs_quspin.py --build-dir build/benchmarks --sizes 12 14 16 18
    python benchmarks/bench_summary.py
"""
from __future__ import annotations
import json
from pathlib import Path


def fmt_us(x):
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x/1000:.2f} ms"
    return f"{x:.1f} us"


def fmt_ms(x):
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x/1000:.2f} s"
    return f"{x:.1f} ms"


def speedup(a, b):
    if a is None or b is None or a == 0:
        return "—"
    return f"{b / a:.2f}x"


def main():
    p = Path("bench_vs_quspin_results.json")
    if not p.exists():
        raise SystemExit(
            "bench_vs_quspin_results.json not found. Run bench_vs_quspin.py first."
        )
    data = json.loads(p.read_text())
    rows = data["rows"]
    threads = data["threads"]

    print(f"# Benchmark vs SOTA peers (Heisenberg PBC ring, threads={threads})\n")
    print("## SpMV throughput (lower is better)\n")
    print("| N | dim | us (complex) | us (real CSR) | QuSpin | scipy.sparse | speedup vs QuSpin | speedup vs scipy |")
    print("|---|----:|-------------:|--------------:|-------:|-------------:|-----------------:|-----------------:|")
    for r in rows:
        ours = r["us_apply_real_us"] or r["us_apply_us"]
        print(
            f"| {r['N']} | {r['dim']:,} | {fmt_us(r['us_apply_us'])} | "
            f"{fmt_us(r['us_apply_real_us'])} | {fmt_us(r['quspin_apply_us'])} | "
            f"{fmt_us(r['scipy_apply_us'])} | {speedup(ours, r['quspin_apply_us'])} | "
            f"{speedup(ours, r['scipy_apply_us'])} |"
        )

    print("\n## Ground-state Lanczos to tol=1e-10 (lower is better)\n")
    print("| N | dim | ours | QuSpin (eigsh ARPACK) | speedup |")
    print("|---|----:|----:|---------------------:|--------:|")
    for r in rows:
        print(
            f"| {r['N']} | {r['dim']:,} | {fmt_ms(r['us_lanczos_ms'])} | "
            f"{fmt_ms(r['quspin_lanczos_ms'])} | "
            f"{speedup(r['us_lanczos_ms'], r['quspin_lanczos_ms'])} |"
        )
    print()


if __name__ == "__main__":
    main()
