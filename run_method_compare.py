#!/usr/bin/env python3
"""Compare E(T), C(T), S(T) across all thermal methods on N=10 Heisenberg."""
from __future__ import annotations
import time, json, sys
import numpy as np
import qed

N    = 10
TMIN = 0.05
TMAX = 20.0
NUMT = 80
bonds = [(i, (i+1)%N) for i in range(N)]

print(f"Building Heisenberg N={N} ...")
H = qed.input.HamiltonianBuilder(N).heisenberg(bonds, J=1.0).to_operator()

# ── exact reference ───────────────────────────────────────────────────────────
print("Running: exact (full diag) ...")
t0 = time.time()
eigs = np.sort(np.array(qed.full_diagonalization(H), float))
print(f"  {len(eigs)} eigenvalues  ({time.time()-t0:.2f}s)")

def exact_thermo(eigs, T):
    T = np.asarray(T, float); beta = 1.0/T
    E0 = eigs.min(); shifted = eigs - E0
    w  = np.exp(-beta[:,None] * shifted[None,:])
    Z  = w.sum(1)
    E  = (w * eigs).sum(1) / Z
    E2 = (w * eigs**2).sum(1) / Z
    Cv = beta**2 * (E2 - E**2)
    lnZ = np.log(Z) - beta*E0
    F  = -T * lnZ
    S  = (E - F) / T
    return E, Cv, S

T_grid = np.geomspace(TMIN, TMAX, NUMT)
Ex, Cx, Sx = exact_thermo(eigs, T_grid)

# ── helper: run one thermal call ──────────────────────────────────────────────
def run(label, **kw):
    print(f"Running: {label} ...", flush=True)
    t0 = time.time()
    try:
        res = qed.thermal(H, T_min=TMIN, T_max=TMAX, num_T=NUMT,
                          use_sz_if_conserved=False, verbose=False, **kw)
        T = np.array(res.temperatures, float)
        E = np.array(res.energy, float)
        C = np.array(res.specific_heat, float)
        S = np.array(res.entropy, float)
        dt = time.time() - t0
        print(f"  done ({dt:.2f}s)  dE={np.max(np.abs(E - np.interp(T, T_grid, Ex))):.4f}")
        return dict(label=label, T=T.tolist(), E=E.tolist(), C=C.tolist(), S=S.tolist(), dt=dt)
    except Exception as e:
        print(f"  FAILED: {e}")
        return None

results = []

# mTPQ – few vs many samples
results.append(run("mTPQ (5 samp)",  method="mTPQ",    num_samples=5,   max_iterations=300, random_seed=42))
results.append(run("mTPQ (20 samp)", method="mTPQ",    num_samples=20,  max_iterations=300, random_seed=42))
results.append(run("mTPQ (80 samp)", method="mTPQ",    num_samples=80,  max_iterations=300, random_seed=42))

# cTPQ
results.append(run("cTPQ (20 samp)", method="cTPQ",    num_samples=20,  max_iterations=300, random_seed=42))

# FTLM – few vs many Krylov vectors
results.append(run("FTLM (m=20)",   method="FTLM",    krylov_dim=20,   num_samples=20,  random_seed=42))
results.append(run("FTLM (m=80)",   method="FTLM",    krylov_dim=80,   num_samples=20,  random_seed=42))
results.append(run("FTLM (m=200)",  method="FTLM",    krylov_dim=200,  num_samples=10,  random_seed=42))

# LTLM
results.append(run("LTLM (m=80)",   method="LTLM",    krylov_dim=80,   num_samples=20,  random_seed=42))

# KPM_DOS
results.append(run("KPM_DOS",        method="KPM_DOS"))

results = [r for r in results if r is not None]

# save
data = dict(
    N=N, T_exact=T_grid.tolist(),
    E_exact=Ex.tolist(), C_exact=Cx.tolist(), S_exact=Sx.tolist(),
    methods=results,
)
with open("method_compare.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"\nSaved method_compare.json  ({len(results)} methods)")
