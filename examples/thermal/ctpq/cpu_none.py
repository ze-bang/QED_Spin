"""thermal | cTPQ | CPU (OpenMP) | full Hilbert (no symmetry)

Canonical Thermal Pure Quantum (Krylov imaginary-time). Twin: ``examples/thermal/ctpq/cpu_none.cpp``.

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
    method="cTPQ",
    T_min=0.1,
    T_max=10.0,
    num_T=8,
    num_samples=4,
    random_seed=0,
    device="cpu",
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
# gs_E    = -3.6511
# T[0]    = 0.1000  E = -3.6455  Cv = 0.2903
# T[mid]  = 5.7571  E = 0.5193  Cv = 0.0241
# T[-1]   = 10.0000  E = 0.6143  Cv = 0.0072
# ===========================================================================
