"""spectral | ground-state DSSF | single GPU (cuBLAS/cuSPARSE) | full Hilbert (no symmetry)

dynamical structure factor S_zz(omega) at three omega points. Twin: ``examples/spectral/ground_state_dssf/gpu_none.cpp``.

Requires: WITH_CUDA build + a visible CUDA device
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = 8
H = heisenberg_chain(N, pbc=True)

obs = qed.Operator(N, 0.5)
for i in range(N):
    obs.add_one_body(qed.OP_SZ, i, 1.0)

omega = np.linspace(-5.0, 5.0, 11)
result = qed.spectral(
    H,
    [obs],
    method="ground_state_cf",
    omega=omega,
    eta=0.1,
    krylov_dim=80,
    device="gpu",
    verbose=False,
)

mid = len(result.omega) // 2
rank0_print(f"S(w=-5.0) = {result.S_real[0]:.6f}")
rank0_print(f"S(w= 0.0) = {result.S_real[mid]:.6f}")
rank0_print(f"S(w= 5.0) = {result.S_real[-1]:.6f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# (filled in by refresh_expected_output.py once the CPU binaries are built)
# ===========================================================================
