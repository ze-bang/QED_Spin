"""spectral | ground-state DSSF from a *finite-T TPQ state* | CPU (OpenMP)

Pillar 3 of the May 2026 "Save and DSSF Upgrades" plan. Demonstrates
the TPQ-to-CF spectral pipeline:

    1. ``qed.thermal(..., method="mTPQ", output_dir, probe_betas=[1.0])``
       persists a TPQ state at beta = 1.
    2. We reload the persisted state vector via ``h5py`` -- the HDF5
       layout uses a compound (real, imag) dtype, so we collapse it
       back to ``numpy.complex128`` before forwarding.
    3. ``qed.spectral(..., method="GroundStateCF",
       initial_state=psi_beta)`` feeds the warm state into the
       continued-fraction kernel. The orchestrator's GroundStateCF
       branch sees a non-empty ``initial_state``, skips the internal
       Lanczos seed step, normalises the user-supplied vector, and
       hands it to ``cf_spectral_kernel`` directly.

The same workflow exists at the C++ level; see twin
``examples/spectral/ground_state_dssf/cpu_from_tpq_state.cpp``.

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

N = 6
H = heisenberg_chain(N, pbc=True)

# -- Step 1: persist an mTPQ state at beta = 1.
outdir = "ed_tpq_to_cf_demo"
if os.path.isdir(outdir):
    shutil.rmtree(outdir)

tres = qed.thermal(
    H,
    method="mTPQ",
    T_min=0.5,
    T_max=4.0,
    num_T=8,
    num_samples=1,
    random_seed=42,
    max_iterations=200,
    device="cpu",
    output_dir=outdir,
    probe_betas=[1.0],
    use_sz_if_conserved=False,
    verbose=False,
)
rank0_print(f"tpq.hdf5_path = {tres.hdf5_path}")

# -- Step 2: reload the persisted state via h5py.
with h5py.File(tres.hdf5_path, "r") as f:
    states = list(f["/tpq/samples/sample_0/states"].keys())
    # Pick the snapshot whose effective beta is closest to 1.0
    # (the kernel rounds each probe beta to the nearest trajectory
    # step, so we discover the actual beta from the listing).
    closest = sorted(
        states,
        key=lambda k: abs(float(k.removeprefix("beta_")) - 1.0),
    )[0]
    raw = np.asarray(f[f"/tpq/samples/sample_0/states/{closest}"][...])
    psi_beta = (raw["real"] + 1j * raw["imag"]).astype(np.complex128)

rank0_print(f"reloaded snapshot       = {closest}")
rank0_print(f"|psi_beta|.shape        = {psi_beta.shape}")

# -- Step 3: feed the warm state into GroundStateCF as the seed.
obs = qed.Operator(N, 0.5)
for i in range(N):
    obs.add_one_body(qed.OP_SZ, i, 1.0)

omega = np.linspace(-5.0, 5.0, 11)
sres = qed.spectral(
    H,
    [obs],
    method="ground_state_cf",
    omega=omega,
    eta=0.1,
    krylov_dim=80,
    device="cpu",
    initial_state=psi_beta,
    verbose=False,
)

mid = len(sres.omega) // 2
rank0_print(f"S(w=-5.0) = {sres.S_real[0]:.6f}")
rank0_print(f"S(w= 0.0) = {sres.S_real[mid]:.6f}")
rank0_print(f"S(w= 5.0) = {sres.S_real[-1]:.6f}")

# === Expected output (smoke-tested by the CI harness; non-zero spectral weight ===
# === at finite omega confirms the TPQ-to-CF pipeline ran end-to-end.) =============
# tpq.hdf5_path = ed_tpq_to_cf_demo/ed_results.h5
# reloaded snapshot       = beta_0.993528
# |psi_beta|.shape        = (64,)
# S(w=-5.0) = 0.000731
# S(w= 0.0) = 0.617431
# S(w= 5.0) = 0.003386
# ===========================================================================
