"""spectral | static thermal | multi-rank multi-GPU (NCCL) | full Hilbert (no symmetry)

static thermodynamic averages at a single T via qed.thermal. Twin: ``examples/spectral/static_thermal/mpi_gpu_none.cpp``.

Requires: WITH_MPI + WITH_CUDA + NCCL; launch via mpirun -n <ranks>
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
    T_min=0.5, T_max=2.0, num_T=4,
    num_samples=8,
    random_seed=0,
    device="mpi_gpu",
    verbose=False,
)

T  = result.temperatures
E  = result.energy
Cv = result.specific_heat
i_T = len(T) // 2  # pick a middle T

rank0_print(f"gs_E    = {result.ground_state_energy:.4f}")
rank0_print(f"T_probe = {T[i_T]:.4f}  E = {E[i_T]:.4f}  Cv = {Cv[i_T]:.4f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# (filled in by refresh_expected_output.py once the CPU binaries are built)
# ===========================================================================
