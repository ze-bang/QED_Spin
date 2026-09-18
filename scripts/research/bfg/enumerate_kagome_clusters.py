#!/usr/bin/env python3
"""Enumerate all clean kagome supercells with n_cells in [nmin, nmax] (HNF), dedupe by
point-group equivalence (identical shortest-vector spectrum + momentum/PG signature),
and report momentum content, |PG|, and estimated largest Sz-sector block size."""
import math, sys, json, argparse
import numpy as np
from edlib.helper_kagome_supercell import KagomeSupercell, enumerate_supercells, bravais_norm2

ap = argparse.ArgumentParser()
ap.add_argument("--nmin", type=int, default=7); ap.add_argument("--nmax", type=int, default=16)
ap.add_argument("--json", default="")
a = ap.parse_args()
rows = []
for nc in range(a.nmin, a.nmax + 1):
    seen = {}
    for L in enumerate_supercells(nc):
        cl = KagomeSupercell(L)
        if cl.min_T2 < 7 or cl.check_clean():
            continue
        # LLL-like reduced shortest two independent vectors for a readable form
        vecs = sorted({(bravais_norm2(*(m1*cl.L[0]+m2*cl.L[1])), tuple(m1*cl.L[0]+m2*cl.L[1]))
                       for m1 in range(-4, 5) for m2 in range(-4, 5) if (m1, m2) != (0, 0)})
        mc = cl.momentum_content(); npg = len(cl.point_group_perms())
        norms = tuple(sorted(n for n, _ in vecs)[:6])
        sig = (norms, tuple(sorted(mc.items())), npg)
        if sig in seen:
            continue
        N = cl.N; nup = N // 2
        order = nc * npg * (2 if N % 2 == 0 else 1)
        rows.append(dict(n_cells=nc, N=N, L=cl.L.tolist(), min_T2=cl.min_T2, PG=npg,
                         G=mc["G"], M=mc["M1"] + mc["M2"] + mc["M3"],
                         M_which="".join(k for k in ("M1", "M2", "M3") if mc[k]),
                         K=mc["K"] + mc["K'"], sector=math.comb(N, nup),
                         block=math.comb(N, nup) / order))
        seen[sig] = True
print(f"{'Nc':>3} {'N':>3} {'L':>18} {'|T|2':>4} {'PG':>3} {'M':>6} {'K':>2} {'Sz-sector':>10} {'~block':>9}")
for r in rows:
    print(f"{r['n_cells']:>3} {r['N']:>3} {str(r['L']):>18} {r['min_T2']:>4} {r['PG']:>3} "
          f"{r['M']:>2}{('('+r['M_which']+')') if r['M'] else '':>4} {r['K']:>2} {r['sector']:>10.2e} {r['block']:>9.1e}")
if a.json:
    json.dump(rows, open(a.json, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
