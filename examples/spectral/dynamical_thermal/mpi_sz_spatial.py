"""spectral | dynamical thermal | MPI (distributed) | U(1)-Sz x cyclic translation

finite-T S_zz(omega, T) at three (T, omega) probe points. Twin: ``examples/spectral/dynamical_thermal/mpi_sz_spatial.cpp``.

Requires: WITH_MPI build + launch via mpirun -n <ranks>
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
    method="ftlm_dynamical",
    T=[0.5, 1.0, 2.0],
    omega=omega,
    eta=0.1,
    krylov_dim=40,
    num_random_vectors=8,
    device="mpi",
    sz=N // 2,
    symmetry=[[(i + 1) % N for i in range(N)]],
    verbose=False,
)

mid = len(result.omega) // 2
rank0_print(f"S(w=-5.0) = {result.S_real[0]:.6f}")
rank0_print(f"S(w= 0.0) = {result.S_real[mid]:.6f}")
rank0_print(f"S(w= 5.0) = {result.S_real[-1]:.6f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# (filled in by refresh_expected_output.py once the CPU binaries are built)
# ===========================================================================
