#!/usr/bin/env python3
"""Gate G1b: 36d feasibility + external Heisenberg references + momentum decoding.

modes:
  decode   : plan_only star table on a cluster (small n_up) -> k_raw -> momentum label
  heis     : NN Heisenberg E0 on a cluster, full space group, vs Lauchli-Sudan-Sorensen
             (arXiv:1103.1159 Table I): 30 (2,1),(-2,4) -13.154318948 ; 36d -15.781555118
  bfg      : BFG Gamma-star block grounds on 36d at a coupling (timing + labels)
"""
import os, sys, time, json, argparse
import numpy as np
import qed
from qed import _core
from edlib.helper_kagome_supercell import KagomeSupercell, build_bfg_operator_supercell

REF = {"30": -13.154318948, "36d": -15.781555118, "3x3": -11.779504985}   # 3x3 = Lauchli 27b
LS = {"24": [[2, 1], [-2, 3]], "30": [[2, 1], [-2, 4]], "36d": [[2, 2], [-2, 4]], "3x3": [[3, 0], [0, 3]]}
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["decode", "heis", "bfg"])
ap.add_argument("--cluster", default="36d")
ap.add_argument("--jpm", type=float, default=0.03)
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--all-stars", action="store_true", help="scan every momentum star (global minimum)")
# One (star, flip-parity) per array task. Flip-extended index: k0_raw + flip*n_irr.
# Without this the Gamma-star job carries every irrep of the star in one process, and the
# little co-group monomials alone are |P| * dim * 20 B = 91 GB at 36d half filling.
ap.add_argument("--k0", default="", help="comma list of flip-extended star indices (overrides Gamma pin)")
ap.add_argument("--irrep", default="", help="comma list of little-co-group irrep indices (only_irrep)")
ap.add_argument("--out", default="")
a = ap.parse_args()
T0 = time.time()
log = lambda *s: print(f"[g1b {time.time()-T0:7.0f}s]", *s, flush=True)
cl = KagomeSupercell(LS[a.cluster])
A = cl.translation_perms()
pg = cl.point_group_perms()
ident = list(range(cl.N))
res = [p for (nm, p) in pg if p != ident]
log(f"cluster {a.cluster} L={cl.L.tolist()} N={cl.N} |T|={len(A)} |PG|={len(pg)} residues={len(res)}")

def decode(chars):
    """k_raw -> (momentum, label, sign) by matching chi(T_t) = exp(2 pi i s q.t)."""
    grid = cl.momentum_grid(); out = {}
    for k, row in enumerate(chars):
        row = np.array(row, complex); hit = None
        for s in (+1, -1):
            for q in grid:
                pred = np.array([np.exp(2j * np.pi * s * (float(q[0]) * t[0] + float(q[1]) * t[1]))
                                 for t in cl.cells])
                if np.max(np.abs(pred - row)) < 1e-8:
                    hit = (q, cl.label_momentum(q), s); break
            if hit: break
        out[k] = hit
    return out

def heis_op():
    nn, _, _ = cl.bond_lists()
    op = qed.Operator(cl.N, 0.5)
    for (i, j) in nn:
        op.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
        op.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
        op.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
    return op

if a.mode == "decode":
    op, _ = build_bfg_operator_supercell(cl.L, 0.03)
    rec = {}
    for nup in (2, 3):
        d = dict(_core.little_group_full_spectrum(op, A, res, n_up=nup, plan_only=True))
        dec = decode(d["irrep_characters"])
        stars = [(int(s["k0"]), int(s["star_size"]), list(map(int, s["members"]))) for s in d["stars"]]
        rec[nup] = (dec, stars)
        log(f"n_up={nup}: {len(d['irrep_characters'])} raw irreps, {len(stars)} stars")
        for (k0, sz, mem) in stars:
            h = dec.get(k0)
            log(f"   star k0={k0:3d} size={sz} members={mem}  -> {h[1] if h else 'UNDECODED'} "
                f"q={tuple(map(str, h[0])) if h else ''} sign={h[2] if h else ''}")
    same = rec[2][1] == rec[3][1] and all(rec[2][0][k] == rec[3][0][k] for k in rec[2][0])
    log(f"star table independent of n_up: {same}")
    if a.out:
        json.dump({str(k): (None if v is None else [str(v[0][0]), str(v[0][1]), v[1], v[2]])
                   for k, v in rec[2][0].items()}, open(a.out, "w"), indent=1)
    sys.exit(0 if same and all(v is not None for v in rec[2][0].values()) else 1)

if a.mode == "heis":
    op = heis_op()
    # star/character table is independent of n_up (checked on 3x3, 24, 30): plan cheaply at n_up=2
    d = dict(_core.little_group_full_spectrum(op, A, res, n_up=2, plan_only=True))
    dec = decode(d["irrep_characters"])
    kG = [k for k, v in dec.items() if v and v[1] == "G"][0]
    nirr = len(d["irrep_characters"])
    if a.k0:
        os.environ["ED_SYM_LG_ONLY_K0"] = a.k0
        which = f"stars {a.k0}"
    elif a.all_stars:
        os.environ.pop("ED_SYM_LG_ONLY_K0", None)
        which = "ALL stars"
    else:
        os.environ["ED_SYM_LG_ONLY_K0"] = f"{kG},{kG + nirr}"   # both spin-flip parities (singlet parity = (-1)^(N/2))
        which = "Gamma star"
    for k in a.k0.split(",") if a.k0 else []:
        lab = dec.get(int(k) % nirr)
        log(f"   pinned k0={k}: {lab[1] if lab else '?'} q={tuple(map(str, lab[0])) if lab else ''} "
            f"flip {'odd' if int(k) >= nirr else 'even'}")
    t = time.time()
    irr = [int(x) for x in a.irrep.split(",")] if a.irrep else []
    g = dict(_core.little_group_block_grounds(op, A, res, n_up=cl.N // 2, dense_max_dim=256,
                                              only_irrep=irr))
    ev = np.array(g["eigenvalues"], float)
    if len(ev) == 0 and irr:
        log(f"irrep(s) {irr} empty in {which} -- no such co-group irrep here (not a failure)")
        if a.out:
            json.dump(dict(cluster=a.cluster, k0=a.k0, irrep=irr, E_min=None, empty=True),
                      open(a.out, "w"), indent=1)
        sys.exit(0)
    if len(ev) == 0:                      # empty return is the silent-failure mode, not a result
        log(f"ABORT: {which} returned no eigenvalues"); sys.exit(3)
    E0 = ev.min()
    log(f"Heisenberg {a.cluster}: {which} block grounds "
        f"{np.round(np.sort(ev)[:6], 9)}  ({time.time()-t:.0f}s)")
    if "k_raw" in g:
        kr = np.array(g["k_raw"], int)
        for k in sorted(set(kr.tolist())):
            base = k % nirr
            lab = dec.get(base)
            log(f"   k_raw={k:3d} ({lab[1] if lab else '?'}, flip {'odd' if k >= nirr else 'even/none'}): "
                f"min E = {ev[kr == k].min():+.9f}")
    if a.out:
        json.dump(dict(cluster=a.cluster, k0=a.k0, irrep=irr, which=which, E_min=float(E0),
                       eigenvalues=sorted(map(float, ev)), ref=REF[a.cluster],
                       seconds=time.time() - t), open(a.out, "w"), indent=1)
    if a.k0:
        # A single star is NOT the gate: the ground state may live in another star (that is
        # exactly how the 30-site run first "failed"). The merge step takes the global minimum.
        log(f"star min = {E0:+.9f}  (ref {REF[a.cluster]}, gap {E0 - REF[a.cluster]:+.3e}) -- "
            f"gate deferred to merge")
        sys.exit(0)
    dif = abs(E0 - REF[a.cluster])
    log(f"GATE vs Lauchli et al. {REF[a.cluster]}: |diff| = {dif:.2e}  {'PASS' if dif < 1e-7 else 'FAIL'}")
    sys.exit(0 if dif < 1e-7 else 1)

if a.mode == "bfg":
    op, _ = build_bfg_operator_supercell(cl.L, a.jpm)
    # star/character table is independent of n_up (checked on 3x3, 24, 30): plan cheaply at n_up=2
    d = dict(_core.little_group_full_spectrum(op, A, res, n_up=2, plan_only=True))
    dec = decode(d["irrep_characters"])
    kG = [k for k, v in dec.items() if v and v[1] == "G"][0]
    nirr = len(d["irrep_characters"])
    os.environ["ED_SYM_LG_ONLY_K0"] = f"{kG},{kG + nirr}"   # both spin-flip parities (singlet parity = (-1)^(N/2))
    t = time.time()
    g = dict(_core.little_group_block_grounds(op, A, res, n_up=cl.N // 2, dense_max_dim=256))
    log(f"BFG 36d Jpm={a.jpm}: Gamma star block grounds in {time.time()-t:.0f}s")
    keys = [k for k in g.keys()]
    log(f"keys: {keys}")
    ev = np.array(g["eigenvalues"], float); o = np.argsort(ev)
    for n in o:
        lab = {k: (g[k][n] if hasattr(g[k], "__len__") and len(g[k]) == len(ev) else None)
               for k in ("k_raw", "irrep", "irrep_dim", "flip_parity", "converged")}
        log(f"   E={ev[n]:+.10f}  dE={ev[n]-ev[o[0]]:+.3e}  {lab}")
    if a.out:
        np.savez(a.out, **{k: np.array(v) for k, v in g.items() if not isinstance(v, (str, dict))})
