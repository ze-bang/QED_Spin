"""spectral | KPM dynamical | CPU (OpenMP) | full Hilbert (no symmetry)

Pillar 4 of the May 2026 "Save and DSSF Upgrades" plan. Demonstrates
the new ``method="kpm_dynamical"`` lane: Chebyshev expansion of
``delta(omega - H)`` against the ground-state seed, evaluated on the
user-supplied omega grid via
``ed::observables::kpm_dynamical_correlator``.

Twin: ``examples/spectral/kpm_dynamical/cpu_none.cpp``.

Requires: (no special requirements; runs on the default CPU build).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import qed

N = 4
H = heisenberg_chain(N, pbc=True)

obs = qed.Operator(N, 0.5)
obs.add_one_body(qed.OP_SZ, 0, 1.0)  # S^z_0 -- single-site probe

omega = np.linspace(-4.0, 4.0, 41)
result = qed.spectral(
    H,
    [obs],
    method="kpm_dynamical",
    omega=omega,
    kpm_moments=400,
    kpm_kernel="Jackson",
    device="cpu",
    verbose=False,
)

mid = len(result.omega) // 2
rank0_print(f"omega.size()      = {len(result.omega)}")
rank0_print(f"S(w=-1.0) approx  = {result.S_real[15]:.4f}")
rank0_print(f"S(w= 0.0) approx  = {result.S_real[mid]:.4f}")
rank0_print(f"S(w=+1.0) approx  = {result.S_real[25]:.4f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# omega.size()      = 41
# S(w=-1.0) approx  = 0.0000
# S(w= 0.0) approx  = 0.0000
# S(w=+1.0) approx  = 2.8864
# ===========================================================================
