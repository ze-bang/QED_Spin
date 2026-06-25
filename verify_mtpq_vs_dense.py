#!/usr/bin/env python3
"""Verify finite-temperature methods (mTPQ focus) against full dense ED.

For every (Hamiltonian x method x symmetry x backend x seed) cell we:
  1. compute the EXACT canonical thermodynamics E(T), C(T), S(T) from the
     full dense spectrum (qed.full_diagonalization -> numpy partition fn),
     evaluated at exactly the temperatures the finite-T method returned;
  2. run the finite-T method via qed.thermal;
  3. compare E, C and (where physically meaningful) S inside the method's
     valid temperature window, with method-appropriate tolerances.

Symmetry axes:
  none       : full Hilbert space (use_sz_if_conserved=False, in-memory)
  sz         : U(1)/Sz per-sector recombination (in-memory)
  spatial    : Z_N translation point group (directory + automorphism_results)
  sz_spatial : U(1) x Z_N simultaneously (directory, Sz on)

Backends: cpu, gpu (gpu skipped for methods with no GPU kernel).

The mTPQ entropy comparison is the headline check for the ln(D)-baseline
fix: before the fix the per-sector free energy lacked the absolute
dimensional normalisation, biasing both S(T) and the cross-sector
recombination of E(T).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import qed

KB = 1.0


# ---------------------------------------------------------------------------
# Exact reference
# ---------------------------------------------------------------------------
def exact_thermo(eigs: np.ndarray, T: np.ndarray):
    """Exact canonical E(T), C(T), S(T) from the full spectrum."""
    eigs = np.asarray(eigs, float)
    T = np.asarray(T, float)
    E0 = eigs.min()
    beta = 1.0 / T
    shifted = eigs[None, :] - E0
    w = np.exp(-beta[:, None] * shifted)              # (nT, dim)
    Z = w.sum(axis=1)
    E = (w * eigs[None, :]).sum(axis=1) / Z
    E2 = (w * (eigs[None, :] ** 2)).sum(axis=1) / Z
    C = beta ** 2 * (E2 - E ** 2)
    lnZ = np.log(Z) - beta * E0                       # ln of true Z
    F = -T * lnZ
    S = (E - F) / T
    return E, C, S


# ---------------------------------------------------------------------------
# Hamiltonians (all N=8, dim 256; full dense diag is trivial)
# ---------------------------------------------------------------------------
def build_hamiltonians(N=8):
    bonds = [(i, (i + 1) % N) for i in range(N)]
    nnn = [(i, (i + 2) % N) for i in range(N)]
    H = {}
    H["heisenberg"] = dict(
        builder=lambda: qed.input.HamiltonianBuilder(N).heisenberg(bonds, J=1.0),
        conserves_sz=True, has_translation=True,
    )
    H["xxz_Jz2"] = dict(
        builder=lambda: qed.input.HamiltonianBuilder(N).xxz(bonds, Jxy=1.0, Jz=2.0),
        conserves_sz=True, has_translation=True,
    )
    H["j1j2"] = dict(
        builder=lambda: (qed.input.HamiltonianBuilder(N)
                         .heisenberg(bonds, J=1.0)
                         .heisenberg(nnn, J=0.5)),
        conserves_sz=True, has_translation=True,
    )
    H["tfim_h1.0"] = dict(
        builder=lambda: qed.input.HamiltonianBuilder(N).transverse_field_ising(
            bonds, J=1.0, h=1.0),
        conserves_sz=False, has_translation=True,
    )
    return N, bonds, H


# ---------------------------------------------------------------------------
# Directory + automorphism_results fixture (Z_N translation)
# ---------------------------------------------------------------------------
def write_directory_with_symmetry(builder_fn, N, root):
    builder_fn().write_directory(root)
    sym = os.path.join(root, "automorphism_results")
    os.makedirs(sym, exist_ok=True)

    def perm(shift):
        return [(i - shift) % N for i in range(N)]

    json.dump([perm(g) for g in range(N)],
              open(os.path.join(sym, "max_clique.json"), "w"))
    json.dump({"generators": [{"permutation": perm(1), "order": N}]},
              open(os.path.join(sym, "minimal_generators.json"), "w"))
    secs = []
    for k in range(N):
        a = -2.0 * np.pi * k / N
        secs.append({"sector_id": k, "quantum_numbers": [k],
                     "phase_factors": [{"real": float(np.cos(a)),
                                        "imag": float(np.sin(a))}]})
    json.dump({"sectors": secs},
              open(os.path.join(sym, "sector_metadata.json"), "w"))
    return root


# ---------------------------------------------------------------------------
# Method configuration: T window, comparison flags and tolerances.
# ---------------------------------------------------------------------------
METHODS = {
    # name : (T_min, T_max, num_T, compare_S, tol_E, tol_C, tol_S, gpu_ok)
    "FTLM":    dict(T=(0.3, 10.0, 14), S=True,  tE=0.05, tC=0.10, tS=0.06, gpu=True),
    "LTLM":    dict(T=(0.02, 0.2, 8),  S=False, tE=0.05, tC=0.15, tS=0.10, gpu=False),
    "KPM_DOS": dict(T=(0.3, 10.0, 14), S=True,  tE=0.08, tC=0.15, tS=0.10, gpu=False),
    "mTPQ":    dict(T=(2.0, 40.0, 12), S=True,  tE=0.08, tC=0.10, tS=0.08, gpu=True),
    "cTPQ":    dict(T=(2.0, 40.0, 12), S=True,  tE=0.10, tC=0.12, tS=0.10, gpu=True),
}


def run_cell(name, ham, eigs, N, method, sym_mode, device, seed,
             directory, log):
    cfg = METHODS[method]
    Tmin, Tmax, numT = cfg["T"]
    kw = dict(method=method, T_min=Tmin, T_max=Tmax, num_T=numT,
              random_seed=seed, device=device, verbose=False)

    # method-specific solver budgets (small but accurate in-window)
    if method in ("mTPQ", "cTPQ"):
        kw.update(num_samples=(20 if sym_mode in ("spatial", "sz_spatial") else 30),
                  max_iterations=(200 if sym_mode in ("spatial", "sz_spatial") else 350))
    elif method == "FTLM":
        kw.update(num_samples=40, krylov_dim=60)
    elif method == "LTLM":
        kw.update(num_samples=30, krylov_dim=64)
    elif method == "KPM_DOS":
        kw.update(kpm_num_moments=512, kpm_num_random_vectors=24)

    # symmetry / source plumbing
    if sym_mode == "none":
        H_in = ham["op"]
        kw.update(use_sz_if_conserved=False)
    elif sym_mode == "sz":
        H_in = ham["op"]
        kw.update(use_sz_if_conserved=True)
    elif sym_mode == "spatial":
        H_in = directory
        kw.update(num_sites=N, spin=0.5, use_symmetry_if_available=True,
                  use_sz_if_conserved=False)
    elif sym_mode == "sz_spatial":
        H_in = directory
        kw.update(num_sites=N, spin=0.5, use_symmetry_if_available=True,
                  use_sz_if_conserved=True)
    else:
        raise ValueError(sym_mode)

    t0 = time.time()
    res = qed.thermal(H_in, **kw)
    dt = time.time() - t0

    T = np.asarray(res.temperatures, float)
    Eg = np.asarray(res.energy, float)
    Cg = np.asarray(res.specific_heat, float)
    Sg = np.asarray(res.entropy, float)
    Ex, Cx, Sx = exact_thermo(eigs, T)

    dE = float(np.max(np.abs(Eg - Ex)))
    dC = float(np.max(np.abs(Cg - Cx)))
    dS = float(np.max(np.abs(Sg - Sx))) if cfg["S"] else float("nan")

    okE = dE <= cfg["tE"]
    okC = dC <= cfg["tC"]
    okS = (dS <= cfg["tS"]) if cfg["S"] else True
    ok = okE and okC and okS
    status = "PASS" if ok else "FAIL"
    sflag = f"dS={dS:.4f}" if cfg["S"] else "dS=  -  "
    line = (f"[{status}] {name:11s} {method:7s} sym={sym_mode:10s} "
            f"dev={device:3s} seed={seed:<5d} "
            f"dE={dE:.4f} dC={dC:.4f} {sflag} ({dt:5.1f}s)")
    log(line)
    return dict(ham=name, method=method, sym=sym_mode, device=device,
                seed=seed, dE=dE, dC=dC, dS=dS, ok=ok, secs=dt,
                tolE=cfg["tE"], tolC=cfg["tC"], tolS=cfg["tS"])


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "verify_mtpq_results.json"
    logf = open(out_path + ".log", "w", buffering=1)

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")

    N, bonds, HAMS = build_hamiltonians(8)
    results = []

    # Precompute operators + exact spectra + symmetry directories.
    for name, h in HAMS.items():
        h["op"] = h["builder"]().to_operator()
        h["eigs"] = np.asarray(qed.full_diagonalization(h["op"]), float)
        d = tempfile.mkdtemp(prefix=f"ed_{name}_")
        write_directory_with_symmetry(h["builder"], N, d)
        h["dir"] = d
        log(f"# {name}: dim={h['eigs'].size}  E0={h['eigs'].min():.6f}  "
            f"conserves_sz={h['conserves_sz']}")

    log("")
    log("=" * 100)
    log("TIER 1 - full method x symmetry x backend matrix on Heisenberg "
        "(1 seed)")
    log("=" * 100)
    name = "heisenberg"
    h = HAMS[name]
    sym_modes = ["none", "sz", "spatial", "sz_spatial"]
    for method in ["FTLM", "LTLM", "KPM_DOS", "mTPQ", "cTPQ"]:
        for sym_mode in sym_modes:
            for device in (["cpu", "gpu"] if METHODS[method]["gpu"] else ["cpu"]):
                # spatial TPQ on GPU is very slow and adds little signal:
                # skip GPU for the spatial/sz_spatial TPQ cells.
                if method in ("mTPQ", "cTPQ") and device == "gpu" \
                        and sym_mode in ("spatial", "sz_spatial"):
                    continue
                try:
                    results.append(run_cell(name, h, h["eigs"], N, method,
                                            sym_mode, device, 11,
                                            h["dir"], log))
                except Exception as e:  # noqa
                    log(f"[ERROR] {name} {method} {sym_mode} {device}: {e}")
                    log(traceback.format_exc())
                json.dump(results, open(out_path, "w"), indent=2)

    log("")
    log("=" * 100)
    log("TIER 2 - mTPQ + FTLM on multiple Hamiltonians, 2 seeds, "
        "{none, sz} x cpu")
    log("=" * 100)
    for name in ["xxz_Jz2", "j1j2", "tfim_h1.0"]:
        h = HAMS[name]
        sym_for = ["none", "sz"] if h["conserves_sz"] else ["none"]
        for method in ["mTPQ", "FTLM"]:
            for sym_mode in sym_for:
                for seed in (11, 23):
                    try:
                        results.append(run_cell(name, h, h["eigs"], N, method,
                                                sym_mode, "cpu", seed,
                                                h["dir"], log))
                    except Exception as e:  # noqa
                        log(f"[ERROR] {name} {method} {sym_mode}: {e}")
                        log(traceback.format_exc())
                    json.dump(results, open(out_path, "w"), indent=2)

    log("")
    log("=" * 100)
    log("TIER 3 - mTPQ on GPU, multiple Hamiltonians, {none, sz}")
    log("=" * 100)
    for name in ["xxz_Jz2", "tfim_h1.0"]:
        h = HAMS[name]
        sym_for = ["none", "sz"] if h["conserves_sz"] else ["none"]
        for sym_mode in sym_for:
            try:
                results.append(run_cell(name, h, h["eigs"], N, "mTPQ",
                                        sym_mode, "gpu", 11, h["dir"], log))
            except Exception as e:  # noqa
                log(f"[ERROR] {name} mTPQ {sym_mode} gpu: {e}")
                log(traceback.format_exc())
            json.dump(results, open(out_path, "w"), indent=2)

    # Summary
    log("")
    log("=" * 100)
    n_pass = sum(1 for r in results if r["ok"])
    log(f"SUMMARY: {n_pass}/{len(results)} cells PASS")
    fails = [r for r in results if not r["ok"]]
    if fails:
        log("FAILURES:")
        for r in fails:
            log(f"  {r['ham']} {r['method']} {r['sym']} {r['device']} "
                f"seed={r['seed']} dE={r['dE']:.4f}(tol {r['tolE']}) "
                f"dC={r['dC']:.4f}(tol {r['tolC']}) dS={r['dS']:.4f}")
    log("=" * 100)
    json.dump(results, open(out_path, "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
