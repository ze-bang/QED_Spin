"""spectral | single expectation | MPI (distributed) | U(1)-Sz, half-filled (Sz=0, n_up=N/2)

ground-state expectation of H via qed.solve. Twin: ``examples/spectral/single_expectation/mpi_sz.cpp``.

Requires: WITH_MPI build + launch via mpirun -n <ranks>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = 8
H = heisenberg_chain(N, pbc=True)

result = qed.solve(
    H,
    num_eigenvalues=2,
    solver="LANCZOS",
    device="mpi",
    sz=N // 2,
    tolerance=1e-10,
    verbose=False,
)

rank0_print(f"<O> = {result.eigenvalues[0]:.10f}  (E_0)")
rank0_print(f"E[0] = {result.eigenvalues[0]:.10f}")
rank0_print(f"E[1] = {result.eigenvalues[1]:.10f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# (filled in by refresh_expected_output.py once the CPU binaries are built)
# ===========================================================================
