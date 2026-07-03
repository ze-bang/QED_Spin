#!/usr/bin/env python3
"""Tour 03 -- dynamics with `qed.spectral`: S(Q, omega) and friends.

The in-memory form takes H + a list of observables. With
symmetry="auto" the run is routed through the U(1) x spatial sector
machinery (ground state solved in its irrep, the probe applied
cross-irrep via the selection rule k_final = k_initial + Q); without
it the full-Hilbert lane runs. Both give the same physics -- the
sector route just scales much further.

Run:  python3 examples/tour/03_dynamics_dssf.py
"""
import cmath

import numpy as np

import qed

N = 12
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

# ---------------------------------------------------------------------------
# 1. Build a momentum-resolved probe: S^z_Q = sum_i e^{iQ r_i} S^z_i / sqrt(N).
#    Op.Sp / Op.Sm probes address the delta-Sz = -/+1 channels (the
#    selection rule is inferred from the operator's terms).
# ---------------------------------------------------------------------------
Q_FRAC = 0.5                      # Q in units of 2*pi => Q = pi
Q = 2.0 * np.pi * Q_FRAC
zb = qed.input.HamiltonianBuilder(N)
for i in range(N):
    zb.add_one_body(qed.input.Op.Sz, i, cmath.exp(1j * Q * i) / np.sqrt(N))
S_zQ = zb.to_operator()

omega = np.linspace(0.0, 4.0, 200)

# ---------------------------------------------------------------------------
# 2. Ground-state DSSF through the sector machinery (T=None ->
#    continued-fraction on the GS). momentum_transfer is Q in
#    fractional units (per generator order) -- it picks the target
#    irrep; sz pins the U(1) pivot block.
# ---------------------------------------------------------------------------
r = qed.spectral(H, [S_zQ], omega=omega, eta=0.05,
                 symmetry="auto", sz=N // 2, momentum_transfer=[Q_FRAC])
print("selection rule:", r.selection_rule_label)
print(f"peak S(Q=pi, w) = {max(np.asarray(r.S_real)):.4f}")

# ---------------------------------------------------------------------------
# 3. Finite-T DSSF: pass T (scalar or list) -- the FTLM cross-irrep
#    lane samples thermal states per source sector and recombines.
# ---------------------------------------------------------------------------
r_T = qed.spectral(H, [S_zQ], omega=omega, eta=0.1, T=[0.5, 1.0],
                   symmetry="auto", sz=N // 2, momentum_transfer=[Q_FRAC],
                   num_random_vectors=20, verbose=False)

# ---------------------------------------------------------------------------
# 4. Knobs that matter:
#      eta            Lorentzian broadening
#      krylov_dim     continued-fraction depth
#      method         "ground_state_cf" (default) | "kpm_dynamical" | ...
#      device         "cpu" | "gpu"
#    Directory form: qed.spectral("runs/mydir", num_sites=N,
#      symmetry={"observable": S_zQ, "momentum_transfer": [0.5]}, ...)
#    covers the same lanes plus multi-Q sweeps
#    (symmetry={"observables": [...], "momentum_points": [...]}).
# ---------------------------------------------------------------------------
r_plain = qed.spectral(H, [S_zQ], omega=omega, eta=0.05, verbose=False)
d = np.max(np.abs(np.asarray(r.S_real) - np.asarray(r_plain.S_real)))
print(f"sector route vs full-Hilbert lane: max|diff| = {d:.2e}")

print("done.")
