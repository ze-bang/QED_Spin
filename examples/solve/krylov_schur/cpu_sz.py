"""solve | KRYLOV_SCHUR | CPU (OpenMP) | U(1)-Sz, half-filled sector (Sz=0, n_up=N/2)

thick-restart Krylov-Schur for 5 lowest eigenvalues. Twin: ``examples/solve/krylov_schur/cpu_sz.cpp``.

Requires: (no special requirements; runs on the default CPU build)
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
    device="cpu",
    sz=N // 2,
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
# E[0] = -3.6510934089
# E[1] = -3.1284190638
# E[2] = -2.6996281483
# E[3] = -2.4587385089
# E[4] = -2.4587385089
# |E0 - E0_Bethe| = 6.22e-15
# ===========================================================================
