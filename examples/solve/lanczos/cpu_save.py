"""solve | LANCZOS | CPU (OpenMP) | full Hilbert (no symmetry)
+ persistent eigenvector save (HDF5)

Demonstrates the May 2026 eigenvector-save contract:

    qed.solve(H, ..., compute_eigenvectors=True, output_dir="<dir>")

writes the eigenvalues + eigenvectors to ``<dir>/ed_results.h5`` and
returns ``result.eigenvectors_path == "<dir>/ed_results.h5"`` (which
mirrors the C++-side ``GroundStateResult::hdf5_path``). Works across
every solver (``LANCZOS`` / ``BLOCK_LANCZOS`` / ``KRYLOV_SCHUR`` /
``FULL``) on the CPU and single-rank GPU lanes. To exercise a different
solver, just change the ``solver=`` argument.

Twin: ``examples/solve/lanczos/cpu_save.cpp``.

Requires: ``h5py`` (used only by this example to verify the round-trip).
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from _shared.common import heisenberg_chain, rank0_print

import h5py
import numpy as np
import qed

N = 8
H = heisenberg_chain(N, pbc=True)

outdir = "ed_save_demo"
if os.path.isdir(outdir):
    shutil.rmtree(outdir)

result = qed.solve(
    H,
    num_eigenvalues=1,
    solver="LANCZOS",
    device="cpu",
    tolerance=1e-12,
    compute_eigenvectors=True,
    output_dir=outdir,
    verbose=False,
)

rank0_print(f"E[0]      = {result.eigenvalues[0]:.10f}")
rank0_print(f"hdf5_path = {result.eigenvectors_path}")

# Round-trip: load the eigenvalues + ground-state vector back from disk.
with h5py.File(result.eigenvectors_path, "r") as f:
    e_loaded   = np.asarray(f["/eigendata/eigenvalues"][...])
    psi_loaded = np.asarray(f["/eigendata/eigenvector_0"][...])
    # `eigenvector_0` is stored as a structured (re, im) dtype; recompose.
    if psi_loaded.dtype.names == ("real", "imag"):
        psi_loaded = psi_loaded["real"] + 1j * psi_loaded["imag"]

rank0_print(f"loaded #eigenvalues  = {e_loaded.size}")
rank0_print(f"loaded |psi|         = {psi_loaded.size}")
rank0_print(f"loaded E[0]          = {e_loaded[0]:.10f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# E[0]      = -3.6510934089
# hdf5_path = ed_save_demo/ed_results.h5
# loaded #eigenvalues  = 1
# loaded |psi|         = 70
# loaded E[0]          = -3.6510934089
# ===========================================================================
# Note: |psi| = 70 (not 256 = 2^8) because qed.solve auto-detects Sz
# conservation for the Heisenberg ring and projects to the half-filled
# Sz=0 sector (8 choose 4 = 70). The C++ twin (`cpu_save.cpp`) does not
# enable the same Sz projection and therefore stores the full 2^8 = 256
# vector; both are correct eigenvectors of H.
