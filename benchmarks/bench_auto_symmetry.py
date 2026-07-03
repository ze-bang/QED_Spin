#!/usr/bin/env python3
"""bench_auto_symmetry.py -- symmetry="auto" vs no symmetry, all verbs.

Measures the wall-time and block-structure effect of the SymmetryEngine
v2 auto-composition (U(1) Sz x spatial group x spin flip x time
reversal) on the three workflow verbs:

  * GS      qed.solve(H, sz=N/2, num_eigenvalues=1)
  * thermal qed.thermal(H, method="mTPQ")
  * DSSF    qed.spectral(H, [S^z_Q], omega=...)   (in-memory route)

Each row is one (verb, N): ``symmetry="off"`` vs ``symmetry="auto"``
on a Heisenberg ring. ``--auto-only-sizes`` adds rows where the
unsymmetrised lane is impractical; ``--verbs`` restricts the verb set
(the thermal flat pool at N >= 20 solves hundreds of small sectors and
is dominated by per-sector fixed costs + HDF5 I/O -- benchmark it at
sizes where that trade is the interesting one).

Usage::

    python3 benchmarks/bench_auto_symmetry.py \
        --sizes 12,16 --auto-only-sizes 20 --verbs gs,thermal,dssf \
        --device cpu --json docs/perf/bench_auto_symmetry.json
"""
from __future__ import annotations

import argparse
import cmath
import json
import time

import numpy as np

import qed


def ring(n):
    b = qed.input.HamiltonianBuilder(n)
    b.heisenberg([(i, (i + 1) % n) for i in range(n)], J=1.0)
    return b.to_operator()


def sz_probe(n, q_frac=0.5):
    q = 2.0 * np.pi * q_frac
    mb = qed.input.HamiltonianBuilder(n)
    for i in range(n):
        mb.add_one_body(qed.input.Op.Sz, i, cmath.exp(1j * q * i) / np.sqrt(n))
    return mb.to_operator()


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


def bench_gs(H, n, device, do_off, rows, verbose=False):
    def gs(sym):
        return qed.solve(H, symmetry=sym, sz=n // 2, num_eigenvalues=1,
                         device=device, verbose=verbose)
    t_auto, r = timed(lambda: gs("auto"))
    dims = [t.sector_dim for t in (r.sector_tags or [])]
    t_off = None
    if do_off:
        t_off, r_off = timed(lambda: gs("off"))
        assert abs(r.eigenvalues[0] - r_off.eigenvalues[0]) < 1e-7, \
            f"GS mismatch at N={n}"
    rows.append(dict(verb="solve (GS, sz=N/2)", N=n, device=device,
                     t_off=t_off, t_auto=t_auto,
                     blocks=len(dims), max_block=max(dims) if dims else 0))


def bench_thermal(H, n, device, do_off, rows, verbose=False):
    def th(sym):
        return qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=16,
                           symmetry=sym, random_seed=3, device=device,
                           verbose=verbose)
    t_auto, r = timed(lambda: th("auto"))
    t_off = None
    if do_off:
        t_off, _ = timed(lambda: th(None))
    per = r.per_sector or []
    rows.append(dict(verb="thermal (mTPQ)", N=n, device=device,
                     t_off=t_off, t_auto=t_auto, blocks=len(per),
                     max_block=max((e.sector_dim for e in per), default=0)))


def bench_dssf(H, n, device, do_off, rows, verbose=False):
    omega = np.linspace(0.0, 4.0, 120)

    def sf(sym):
        kw = dict(omega=omega, eta=0.1, verbose=verbose)
        if sym is not None:
            kw.update(symmetry=sym, sz=n // 2, momentum_transfer=[0.5])
        return qed.spectral(H, [sz_probe(n)], **kw)
    t_auto, r = timed(lambda: sf("auto"))
    t_off = None
    if do_off:
        t_off, r_off = timed(lambda: sf(None))
        a = np.asarray(r.S_real).ravel()
        b = np.asarray(r_off.S_real).ravel()
        # Sanity only: both lanes are iterative Lanczos CF and converge
        # at different rates; exact small-N parity is pinned by
        # python/tests/test_auto_symmetry.py (1e-8).
        assert np.max(np.abs(a - b)) < 5e-2 * max(1.0, np.max(np.abs(b))), \
            f"DSSF mismatch at N={n}"
    rows.append(dict(verb="spectral (DSSF S^z_Q)", N=n, device=device,
                     t_off=t_off, t_auto=t_auto, blocks=None,
                     max_block=None))


BENCHES = {"gs": bench_gs, "thermal": bench_thermal, "dssf": bench_dssf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="12,16")
    ap.add_argument("--auto-only-sizes", default="")
    ap.add_argument("--verbs", default="gs,thermal,dssf")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    verbs = [v.strip() for v in args.verbs.split(",") if v.strip()]
    rows: list[dict] = []
    for n in [int(x) for x in args.sizes.split(",") if x]:
        H = ring(n)
        for v in verbs:
            BENCHES[v](H, n, args.device, True, rows)
    for n in [int(x) for x in args.auto_only_sizes.split(",") if x]:
        H = ring(n)
        for v in verbs:
            BENCHES[v](H, n, args.device, False, rows)

    print("\n| verb | N | device | t_off [s] | t_auto [s] | speedup "
          "| blocks | max block dim |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        sp = f"{r['t_off'] / r['t_auto']:.1f}x" if r["t_off"] else "--"
        toff = f"{r['t_off']:.2f}" if r["t_off"] else "--"
        blocks = r["blocks"] if r["blocks"] is not None else "--"
        maxb = r["max_block"] if r["max_block"] is not None else "--"
        print(f"| {r['verb']} | {r['N']} | {r['device']} | {toff} "
              f"| {r['t_auto']:.2f} | {sp} | {blocks} | {maxb} |")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
