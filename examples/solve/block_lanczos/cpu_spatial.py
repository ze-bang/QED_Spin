"""solve | BLOCK_LANCZOS | CPU (OpenMP) | cyclic translation group Z_N (spatial symmetry)

block-Lanczos for 4 lowest eigenvalues (BLAS-3 path). Twin: ``examples/solve/block_lanczos/cpu_spatial.cpp``.

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
    num_eigenvalues=4,
    solver="BLOCK_LANCZOS",
    device="cpu",
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
# E[0] = -3.6043724189
# E[1] = -2.9334717436
# E[2] = -2.4587385089
# E[3] = -2.4587385089
# |E0 - E0_Bethe| = 4.67e-02
# ===========================================================================
