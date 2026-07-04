#!/usr/bin/env python3
"""Tour 01 -- ground states with `qed.solve`: every real knob.

One script, one verb. Build a Hamiltonian, then peel the onion:
plain solve -> symmetry="auto" -> per-symmetry toggles -> solver /
device / eigenvector knobs -> per-sector attribution.

Run:  python3 examples/tour/01_ground_state.py
"""
import numpy as np

import qed

N = 12

# ---------------------------------------------------------------------------
# 1. Build a Hamiltonian. HamiltonianBuilder has model shortcuts
#    (heisenberg / xxz / transverse_field_ising / zeeman / ...) and raw
#    add_one_body / add_two_body(Op.Sp|Sm|Sz, site, ...) escapes.
# ---------------------------------------------------------------------------
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)   # ring
H = b.to_operator()

# ---------------------------------------------------------------------------
# 2. The one-liner: maximal block diagonalisation, automatically.
#    symmetry="auto" runs the automorphism search internally; U(1) Sz,
#    spin flip and time reversal auto-detect independently. sz=N//2
#    pins the half-filling U(1) block (auto_sz=True would scan).
# ---------------------------------------------------------------------------
r = qed.solve(H, symmetry="auto", sz=N // 2, num_eigenvalues=4)
print(f"E0 = {r.eigenvalues[0]:.10f}   (lowest {len(r.eigenvalues)} pooled)")

# ---------------------------------------------------------------------------
# 3. Per-symmetry toggles: "auto" | "on" | "off" | "require".
#    "on" REPORTS -- it confirms the detection, or warns and continues
#    without the symmetry when your Hamiltonian lacks it. "require"
#    throws instead of degrading. Mix and match freely.
# ---------------------------------------------------------------------------
r = qed.solve(H, symmetry="auto", sz=N // 2, num_eigenvalues=1,
              spin_flip="on",        # flip transport + (k,+/-) projection
              time_reversal="on")    # solve k, copy spectrum to -k

# What does my Hamiltonian actually have?
print("detected:", dict(qed._core.detect_hamiltonian_symmetries(H)))
report = qed.find_symmetries(H, verbose=False)
print(f"U(1) Sz: {report.has_u1_sz}, spatial |G| = "
      f"{report.full_set.group_size if report.full_set else 1}")

# ---------------------------------------------------------------------------
# 4. Solver / device / convergence knobs.
#    solver: "lanczos" (default) | "block_lanczos" | "krylov_schur" | "full"
#    device: "auto" (default) | "cpu" | "gpu" | "mpi" | "mpi_gpu"
# ---------------------------------------------------------------------------
r = qed.solve(H, symmetry="auto", sz=N // 2, num_eigenvalues=6,
              solver="full",              # exact dense per sector
              device="cpu",
              tolerance=1e-12, max_iterations=500)

# Per-sector attribution: which irrep each eigenvalue came from.
for tag, eigs in zip(r.sector_tags[:4], r.eigenvalues_per_sector[:4]):
    print(f"  sector k-index {tag.sector_index:3d}  dim {tag.sector_dim:5d} "
          f" n_up {tag.n_up}  lowest {eigs[0]:+.6f}")

# ---------------------------------------------------------------------------
# 5. Eigenvectors + saving. compute_eigenvectors=True keeps the states
#    (flip projection then auto-disables -- projected eigenvectors are
#    not orbit-reconstructable); output_dir writes per-sector HDF5.
# ---------------------------------------------------------------------------
r = qed.solve(H, sz=N // 2, num_eigenvalues=1,
              compute_eigenvectors=True, output_dir="")   # "" = don't save

print("done.")
