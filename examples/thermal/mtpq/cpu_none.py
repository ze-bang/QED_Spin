"""thermal | mTPQ | CPU (OpenMP) | full Hilbert (no symmetry)

Micro-canonical Thermal Pure Quantum (Taylor truncation). Twin: ``examples/thermal/mtpq/cpu_none.cpp``.

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
    method="mTPQ",
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
# gs_E    = -3.6506
# T[0]    = 0.1000  E = -3.6393  Cv = 0.6144
# T[mid]  = 5.7571  E = 0.1105  Cv = 0.0344
# T[-1]   = 10.0000  E = 0.1946  Cv = 0.0114
# ===========================================================================
