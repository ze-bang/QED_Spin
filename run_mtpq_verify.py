#!/usr/bin/env python3
"""Focused mTPQ vs dense ED verification across 4 symmetry modes.

Symmetry modes: none, sz, spatial (point group), sz_spatial (Sz + point group).
Hamiltonians: heisenberg, xxz_Jz2, j1j2 (all N=8, dim=256).
Compares E(T), C(T), S(T) against exact canonical partition function.
"""
from __future__ import annotations

import json, os, sys, tempfile, time
import numpy as np
import qed

KB = 1.0

# ── Exact reference ──────────────────────────────────────────────────────────
def exact_thermo(eigs, T):
    eigs = np.asarray(eigs, float); T = np.asarray(T, float)
    E0 = eigs.min(); beta = 1.0 / T
    shifted = eigs[None, :] - E0
    w = np.exp(-beta[:, None] * shifted)
    Z = w.sum(axis=1)
    E  = (w * eigs[None, :]).sum(axis=1) / Z
    E2 = (w * eigs[None, :] ** 2).sum(axis=1) / Z
    C  = beta**2 * (E2 - E**2)
    lnZ = np.log(Z) - beta * E0
    F  = -T * lnZ
    S  = (E - F) / T
    return E, C, S

# ── Hamiltonians ─────────────────────────────────────────────────────────────
def build_hamiltonians(N=8):
    bonds = [(i, (i+1)%N) for i in range(N)]
    nnn   = [(i, (i+2)%N) for i in range(N)]
    return {
        "heisenberg": dict(
            builder=lambda: qed.input.HamiltonianBuilder(N).heisenberg(bonds, J=1.0),
            conserves_sz=True),
        "xxz_Jz2": dict(
            builder=lambda: qed.input.HamiltonianBuilder(N).xxz(bonds, Jxy=1.0, Jz=2.0),
            conserves_sz=True),
        "j1j2": dict(
            builder=lambda: (qed.input.HamiltonianBuilder(N)
                             .heisenberg(bonds, J=1.0)
                             .heisenberg(nnn, J=0.5)),
            conserves_sz=True),
    }, N

# ── Write symmetry directory ──────────────────────────────────────────────────
def write_dir(builder_fn, N, root):
    builder_fn().write_directory(root)
    sym = os.path.join(root, "automorphism_results"); os.makedirs(sym, exist_ok=True)
    def perm(shift): return [(i-shift)%N for i in range(N)]
    json.dump([perm(g) for g in range(N)],
              open(os.path.join(sym, "max_clique.json"), "w"))
    json.dump({"generators": [{"permutation": perm(1), "order": N}]},
              open(os.path.join(sym, "minimal_generators.json"), "w"))
    secs = []
    for k in range(N):
        a = -2.0*np.pi*k/N
        secs.append({"sector_id": k, "quantum_numbers": [k],
                     "phase_factors": [{"real": float(np.cos(a)),
                                        "imag": float(np.sin(a))}]})
    json.dump({"sectors": secs}, open(os.path.join(sym, "sector_metadata.json"), "w"))
    return root

# ── Run one cell ──────────────────────────────────────────────────────────────
# mTPQ is most accurate at high T (T >= 2). Use T window [2, 40] for 12 points.
TMIN, TMAX, NUMT = 2.0, 40.0, 12
TOL = dict(tE=0.08, tC=0.10, tS=0.08)

def run_cell(ham_name, ham, eigs, N, sym_mode, seed):
    kw = dict(method="mTPQ", T_min=TMIN, T_max=TMAX, num_T=NUMT,
              random_seed=seed, device="cpu", verbose=False,
              num_samples=(20 if sym_mode in ("spatial","sz_spatial") else 30),
              max_iterations=(200 if sym_mode in ("spatial","sz_spatial") else 350))
    if sym_mode == "none":
        kw.update(use_sz_if_conserved=False); H_in = ham["op"]
    elif sym_mode == "sz":
        kw.update(use_sz_if_conserved=True); H_in = ham["op"]
    elif sym_mode == "spatial":
        H_in = ham["dir"]
        kw.update(num_sites=N, spin=0.5, use_symmetry_if_available=True,
                  use_sz_if_conserved=False)
    elif sym_mode == "sz_spatial":
        H_in = ham["dir"]
        kw.update(num_sites=N, spin=0.5, use_symmetry_if_available=True,
                  use_sz_if_conserved=True)

    t0 = time.time()
    res = qed.thermal(H_in, **kw)
    dt = time.time() - t0

    T   = np.asarray(res.temperatures, float)
    Eg  = np.asarray(res.energy, float)
    Cg  = np.asarray(res.specific_heat, float)
    Sg  = np.asarray(res.entropy, float)
    Ex, Cx, Sx = exact_thermo(eigs, T)

    dE = float(np.max(np.abs(Eg - Ex)))
    dC = float(np.max(np.abs(Cg - Cx)))
    dS = float(np.max(np.abs(Sg - Sx)))
    ok = dE <= TOL["tE"] and dC <= TOL["tC"] and dS <= TOL["tS"]
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {ham_name:11s} sym={sym_mode:10s} seed={seed}  "
          f"dE={dE:.4f}(≤{TOL['tE']})  dC={dC:.4f}(≤{TOL['tC']})  "
          f"dS={dS:.4f}(≤{TOL['tS']})  ({dt:.1f}s)", flush=True)
    return dict(ham=ham_name, sym=sym_mode, seed=seed,
                dE=dE, dC=dC, dS=dS, ok=ok, secs=dt,
                T=T.tolist(), Eg=Eg.tolist(), Cg=Cg.tolist(), Sg=Sg.tolist(),
                Ex=Ex.tolist(), Cx=Cx.tolist(), Sx=Sx.tolist())

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "mtpq_vs_dense.json"
    HAMS, N = build_hamiltonians(8)

    print("Building operators and exact spectra...")
    for name, h in HAMS.items():
        h["op"]   = h["builder"]().to_operator()
        h["eigs"] = np.asarray(qed.full_diagonalization(h["op"]), float)
        d = tempfile.mkdtemp(prefix=f"ed_{name}_")
        write_dir(h["builder"], N, d)
        h["dir"] = d
        print(f"  {name}: dim={h['eigs'].size}  E0={h['eigs'].min():.6f}")

    SYM_MODES = ["none", "sz", "spatial", "sz_spatial"]
    results = []
    print("\n" + "="*80)
    print("mTPQ vs dense ED  (N=8, T in [2, 40])")
    print("="*80)
    import traceback
    for name in HAMS:
        h = HAMS[name]
        for sym_mode in SYM_MODES:
            if sym_mode == "sz" and not h["conserves_sz"]:
                continue
            for seed in (11, 23):
                try:
                    r = run_cell(name, h, h["eigs"], N, sym_mode, seed)
                    results.append(r)
                except Exception as e:
                    print(f"[ERROR] {name} {sym_mode} seed={seed}: {e}")
                    traceback.print_exc()
                json.dump(results, open(out, "w"), indent=2)

    print("\n" + "="*80)
    n_pass = sum(1 for r in results if r["ok"])
    print(f"SUMMARY: {n_pass}/{len(results)} cells PASS")
    fails = [r for r in results if not r["ok"]]
    if fails:
        print("FAILURES:")
        for r in fails:
            print(f"  {r['ham']} sym={r['sym']} seed={r['seed']}  "
                  f"dE={r['dE']:.4f}  dC={r['dC']:.4f}  dS={r['dS']:.4f}")
    print("="*80)
    print(f"Results saved to {out}")

if __name__ == "__main__":
    main()
