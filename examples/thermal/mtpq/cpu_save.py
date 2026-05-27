"""thermal | mTPQ | CPU (OpenMP) | full Hilbert (no symmetry)
+ persistent trajectory + state-vector snapshots at probe-betas (HDF5)

Pillar 1 of the May 2026 "Save and DSSF Upgrades" plan. Demonstrates::

    qed.thermal(H, ..., method="mTPQ",
                output_dir="<dir>",
                probe_betas=[beta_1, beta_2, ...])

writes per-sample trajectories and host-side state vectors at the
kernel-step beta closest to each user-requested probe beta into
``<dir>/ed_results.h5`` and surfaces the path via ``result.hdf5_path``.

The state snapshots land at ``/tpq/samples/sample_<s>/states/beta_<b>``
and can be reloaded via h5py for the TPQ-to-CF spectral pipeline (see
``examples/spectral/ground_state_dssf/cpu_from_tpq_state.py``).

Twin: ``examples/thermal/mtpq/cpu_save.cpp``.

Requires: ``h5py``.
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

outdir = "ed_thermal_save_demo"
if os.path.isdir(outdir):
    shutil.rmtree(outdir)

result = qed.thermal(
    H,
    method="mTPQ",
    T_min=0.1,
    T_max=10.0,
    num_T=8,
    num_samples=1,
    random_seed=42,
    max_iterations=200,
    device="cpu",
    output_dir=outdir,
    probe_betas=[1.0, 5.0],
    use_sz_if_conserved=False,
    verbose=False,
)

rank0_print(f"hdf5_path = {result.hdf5_path}")
rank0_print(f"gs_E      = {result.ground_state_energy:.6f}")

# Reload via h5py to verify the on-disk artefact carries the expected
# structure: per-sample trajectory rows + state vectors at the
# kernel-step betas closest to the user-requested probe betas.
with h5py.File(result.hdf5_path, "r") as f:
    sample0 = f["/tpq/samples/sample_0"]
    traj    = sample0["thermodynamics"][...]  # (n_steps, 5)
    states  = list(sample0["states"].keys())
    rank0_print(f"trajectory rows         = {traj.shape[0]}")
    rank0_print(f"snapshot datasets       = {sorted(states)}")
    psi5_name = sorted(states, key=lambda k: abs(
        float(k.removeprefix("beta_")) - 5.0))[0]
    # The HDF5 layout uses a compound (real, imag) datatype; collapse to
    # std::complex<double> on the Python side.
    raw = np.asarray(sample0["states"][psi5_name][...])
    psi5 = raw["real"] + 1j * raw["imag"]
    norm2 = float(np.vdot(psi5, psi5).real)
    rank0_print(f"|psi(beta~=5)|.shape    = {psi5.shape}")
    rank0_print(f"<psi|psi>               = {norm2:.6f}")

# === Expected output (deterministic; captured on the CI reference runner) ===
# hdf5_path = ed_thermal_save_demo/ed_results.h5
# gs_E      = -3.637107
# trajectory rows         = 201
# snapshot datasets       = ['beta_1.004994', 'beta_5.001079']
# |psi(beta~=5)|.shape    = (256,)
# <psi|psi>               = 1.000000
# ===========================================================================
