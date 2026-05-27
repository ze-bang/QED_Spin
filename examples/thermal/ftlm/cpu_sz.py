"""thermal | FTLM | CPU (OpenMP) | U(1)-Sz auto-decomposition

Finite-Temperature Lanczos: random vectors x Lanczos. Twin: ``examples/thermal/ftlm/cpu_sz.cpp``.

Requires: (no special requirements; runs on the default CPU build)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = 8
H = heisenberg_chain(N, pbc=True)

result = qed.thermal(
    H,
    method="FTLM",
    T_min=0.1,
    T_max=10.0,
    num_T=8,
    num_samples=8,
    random_seed=0,
    device="cpu",
    use_sz_if_conserved=True,
    verbose=False,
)

T  = result.temperatures
E  = result.energy
Cv = result.specific_heat
mid = len(T) // 2

rank0_print(f"gs_E    = {result.ground_state_energy:.4f}")
rank0_print(f"T[0]    = {T[0]:.4f}  E = {E[0]:.4f}  Cv = {Cv[0]:.4f}")
rank0_print(f"T[mid]  = {T[mid]:.4f}  E = {E[mid]:.4f}  Cv = {Cv[mid]:.4f}")
rank0_print(f"T[-1]   = {T[-1]:.4f}  E = {E[-1]:.4f}  Cv = {Cv[-1]:.4f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# gs_E    = -3.6484
# T[0]    = 0.1000  E = -3.6420  Cv = 0.4751
# T[mid]  = 5.7571  E = -1.0500  Cv = 0.6946
# T[-1]   = 10.0000  E = -0.1798  Cv = 0.0161
# ===========================================================================
