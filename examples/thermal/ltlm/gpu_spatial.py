"""thermal | LTLM | single GPU (cuBLAS/cuSPARSE) | cyclic translation group Z_N (spatial symmetry)

Low-Temperature Lanczos Method. Twin: ``examples/thermal/ltlm/gpu_spatial.cpp``.

Requires: WITH_CUDA build + a visible CUDA device
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
    method="LTLM",
    T_min=0.1,
    T_max=10.0,
    num_T=8,
    num_samples=8,
    random_seed=0,
    device="gpu",
    use_symmetry_if_available=True,
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
# gs_E    = -3.6510934089
# T[0]    = 0.10   E = ...   Cv = ...
# T[mid]  = ...    E = ...   Cv = ...
# T[-1]   = 10.00  E = ...   Cv = ...
# (filled in by refresh_expected_output.py)
# ===========================================================================
