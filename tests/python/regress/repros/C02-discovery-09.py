# AUDIT-ID: C02-discovery-09
# DEVICE: cpu
# SECONDS: 60
"""Claim: the engine's time reversal is complex conjugation K in the Sz basis
(time_reversal.h:42-55, lg_engine.cpp:73-85), not the physical antiunitary
Theta = (prod_i i sigma^y_i) K. A Heisenberg + D_z ring is Theta-invariant (it is a bilinear
spin model) but has complex coefficients, so time_reversal='require' raises and 'auto' never
folds. Dense check of Theta H Theta^-1 = H from H.apply."""

import signal
import numpy as np
import qed

signal.alarm(200)
N = 8
bonds = [(i, (i + 1) % N) for i in range(N)]
b = qed.input.HamiltonianBuilder(N)
b.heisenberg(bonds, 1.0)
b.dm(bonds, [[0.0, 0.0, 0.2]] * N)
H = b.to_operator()

d = 1 << N
M = np.zeros((d, d), complex)
for j in range(d):
    e = np.zeros(d, complex)
    e[j] = 1.0
    M[:, j] = np.asarray(H.apply(e))
herm = float(np.max(np.abs(M - M.conj().T)))
# U = prod_i (i sigma^y_i): |s> -> sign(s) |s ^ all-ones>, sign = prod over sites of (+1 or -1)
mask = d - 1
U = np.zeros((d, d))
for s in range(d):
    ones = bin(s).count("1")
    U[s ^ mask, s] = (-1.0) ** (N - ones)
theta_res = float(np.max(np.abs(U @ M.conj() @ U.T - M)))
k_res = float(np.max(np.abs(M.conj() - M)))

T = [(i + 1) % N for i in range(N)]
try:
    qed.eigs(H, 1, sym=qed.Symmetry(spatial=[T], point_group=False, time_reversal="require"))
    err = None
except Exception as e:
    err = f"{type(e).__name__}: {str(e)[:110]}"
info = (
    f"hermiticity {herm:.1e}; ||Theta H Theta^-1 - H||={theta_res:.1e}; ||K H K - H||={k_res:.2e}; "
    f"time_reversal='require' -> {err!r}"
)
if theta_res < 1e-12 and k_res > 1e-3 and err is not None:
    print("REPRO: CONFIRMED " + info)
elif err is None:
    print("REPRO: NOT_REPRODUCED " + info)
else:
    print("REPRO: INCONCLUSIVE " + info)
