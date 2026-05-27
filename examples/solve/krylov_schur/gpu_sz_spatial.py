"""solve | KRYLOV_SCHUR | single GPU (cuBLAS/cuSPARSE) | U(1)-Sz x cyclic translation (Sz=0)

thick-restart Krylov-Schur for 5 lowest eigenvalues. Twin: ``examples/solve/krylov_schur/gpu_sz_spatial.cpp``.

Requires: WITH_CUDA build + a visible CUDA device
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import bethe_E0, heisenberg_chain, rank0_print

import qed

N = 8
H = heisenberg_chain(N, pbc=True)

result = qed.solve(
    H,
    num_eigenvalues=5,
    solver="KRYLOV_SCHUR",
    device="gpu",
    sz=N // 2,
    symmetry=[[(i + 1) % N for i in range(N)]],
    tolerance=1e-10,
    verbose=False,
)

for k, ek in enumerate(result.eigenvalues):
    rank0_print(f"E[{k}] = {ek:.10f}")

E0 = result.eigenvalues[0]
Eref = bethe_E0(N)
if math.isfinite(Eref):
    rank0_print(f"|E0 - E0_Bethe| = {abs(E0 - Eref):.2e}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# E[0] = -3.6510934089   (filled in by refresh_expected_output.py)  (captured on the gpu reference runner)
# E[1] = ...
# E[2] = ...
# E[3] = ...
# E[4] = ...
# |E0 - E0_Bethe| ~ 1e-10
# ===========================================================================
