#!/usr/bin/env python3
"""Tour 05 -- finite-temperature DSSF from mTPQ states.

A single TPQ state |psi_beta> is a thermal-ensemble typical state: for
large enough Hilbert dims, <psi_beta| A |psi_beta> approximates the
canonical average at that beta. That makes it a cheap seed for
FINITE-TEMPERATURE dynamics: run mTPQ once, persist the state at the
inverse temperatures you care about, and hand each snapshot to the
continued-fraction spectral kernel as its starting vector -- S(Q, w)
at T = 1/beta without ever building the full thermal density matrix.

Pipeline:
  1. qed.thermal(method="mTPQ", probe_betas=[...], output_dir=...)
     persists a state snapshot at each requested beta.
  2. Reload the snapshot with h5py (compound (real, imag) dtype).
  3. qed.spectral(..., initial_state=psi_beta) seeds GroundStateCF
     with the warm state instead of the ground state.

Run:  python3 examples/tour/05_tpq_dssf.py     (requires h5py)
"""
import cmath
import os
import shutil

import h5py
import numpy as np

import qed

N = 10
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# ---------------------------------------------------------------------------
# 1. One mTPQ run, snapshots at beta = 0.5 and 2.0 (T = 2 and T = 0.5).
#    use_sz_if_conserved=False keeps the state in the full 2^N basis so
#    the spectral kernel below can consume it directly.
# ---------------------------------------------------------------------------
outdir = "ed_tour_tpq_dssf"
if os.path.isdir(outdir):
    shutil.rmtree(outdir)

tres = qed.thermal(H, method="mTPQ",
                   T_min=0.4, T_max=5.0, num_T=12,
                   num_samples=1, random_seed=42, max_iterations=300,
                   probe_betas=[0.5, 2.0],
                   use_sz_if_conserved=False,
                   output_dir=outdir, device="cpu", verbose=False)
print(f"mTPQ E(T_min) = {tres.energy[0]:.6f}   snapshots in {tres.hdf5_path}")

# ---------------------------------------------------------------------------
# 2. Reload the snapshots. The kernel rounds each probe beta to the
#    nearest trajectory step; discover the effective betas from the file.
# ---------------------------------------------------------------------------
snapshots = {}
with h5py.File(tres.hdf5_path, "r") as f:
    for key in f["/tpq/samples/sample_0/states"]:
        raw = np.asarray(f[f"/tpq/samples/sample_0/states/{key}"][...])
        beta = float(key.removeprefix("beta_"))
        snapshots[beta] = (raw["real"] + 1j * raw["imag"]).astype(complex)
print("snapshot betas:", [f"{b:.3f}" for b in sorted(snapshots)])

# ---------------------------------------------------------------------------
# 3. S^z_{Q=pi}(w) at each temperature: seed the continued fraction
#    with the warm TPQ state. Colder state (larger beta) concentrates
#    the weight toward the low-w magnon response.
# ---------------------------------------------------------------------------
zb = qed.input.HamiltonianBuilder(N)
for i in range(N):
    zb.add_one_body(qed.input.Op.Sz, i, cmath.exp(1j * np.pi * i) / np.sqrt(N))
S_zQ = zb.to_operator()

omega = np.linspace(0.0, 4.0, 160)
for beta in sorted(snapshots):
    r = qed.spectral(H, [S_zQ], method="ground_state_cf",
                     omega=omega, eta=0.1, krylov_dim=120,
                     initial_state=snapshots[beta],
                     device="cpu", verbose=False)
    S = np.asarray(r.S_real)
    pk = int(np.argmax(S))
    print(f"beta = {beta:5.3f}  (T = {1.0/beta:4.2f}):  "
          f"peak S(Q=pi) = {S[pk]:.4f} at w = {omega[pk]:.3f},  "
          f"integrated = {np.trapezoid(S, omega):.4f}")

shutil.rmtree(outdir, ignore_errors=True)
print("done.")
