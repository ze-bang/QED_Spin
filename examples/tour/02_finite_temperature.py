#!/usr/bin/env python3
"""Tour 02 -- finite temperature with `qed.thermal`: every real knob.

Methods: mTPQ (microcanonical TPQ, default choice for large N),
cTPQ (canonical), FTLM / LTLM (stochastic Lanczos), KPM_DOS, and exact
ED for small blocks. All of them run per (Sz, irrep) sector when
symmetry is on, and the flat pool recombines Z_s(beta) exactly --
including the spin-flip mirror (solve n_up <= N/2, copy to N - n_up)
and time-reversal pairing (solve k, copy to -k).

Run:  python3 examples/tour/02_finite_temperature.py
"""
import numpy as np

import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# ---------------------------------------------------------------------------
# 1. The one-liner: mTPQ over ALL symmetry blocks, auto-composed.
# ---------------------------------------------------------------------------
r = qed.thermal(H, method="mTPQ",
                T_min=0.05, T_max=10.0, num_T=48,
                symmetry="auto",          # spatial group found internally
                spin_flip="on",           # report + exploit (or degrade)
                time_reversal="on",
                random_seed=7)
print(f"E(T_min) = {r.energy[0]:.6f}   C(T) peak = {max(r.specific_heat):.4f}")
print(f"sectors solved/pooled: {len(r.per_sector)}")

# Per-sector entries: (n_up, sector_dim, free_energy, ...) -- the
# mirrored / TR-copied sectors are present with correct multiplicity.
ups = sorted({e.n_up for e in r.per_sector})
print(f"n_up coverage: {ups[0]}..{ups[-1]} "
      f"(flip transport solved only n_up <= {N//2})")

# ---------------------------------------------------------------------------
# 2. Method zoo. Same call shape; the method-specific knobs are:
#      mTPQ/cTPQ : num_samples, tpq_measure_beta_min/max
#      FTLM/LTLM : num_samples, ftlm_krylov_dim / ltlm_krylov_dim
#      KPM_DOS   : kpm_num_moments, kpm_num_random_vectors
# ---------------------------------------------------------------------------
r_ftlm = qed.thermal(H, method="FTLM", T_min=0.1, T_max=10.0, num_T=24,
                     symmetry="auto", num_samples=20, ftlm_krylov_dim=80,
                     random_seed=7, verbose=False)

# ---------------------------------------------------------------------------
# 3. U(1) window: restrict the Sz pool (e.g. around half filling), or
#    disable the Sz decomposition entirely with use_sz_if_conserved.
# ---------------------------------------------------------------------------
r_win = qed.thermal(H, method="mTPQ", T_min=0.2, T_max=5.0, num_T=16,
                    symmetry="auto", sz_min=N // 2 - 1, sz_max=N // 2 + 1,
                    random_seed=7, verbose=False)

# ---------------------------------------------------------------------------
# 4. Backend: device="cpu" | "gpu" pins the lane ("auto" picks GPU when
#    the block dims warrant it). Everything above works on either.
# ---------------------------------------------------------------------------
r_cpu = qed.thermal(H, method="mTPQ", T_min=0.5, T_max=5.0, num_T=8,
                    symmetry="auto", device="cpu", random_seed=7,
                    verbose=False)

print("done.")
