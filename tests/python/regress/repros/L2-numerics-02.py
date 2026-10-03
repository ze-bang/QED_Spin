# AUDIT-ID: L2-numerics-02
# DEVICE: cpu
# SECONDS: 120
"""Claim: qed.dynamics(T=None) averages over every level within degeneracy_tol = 1e-8 of E0,
an ABSOLUTE window in the units of H (lg_sectors_dynamics.cpp:161-184). Rescaling H (same
physics in other units) changes which states form the ground manifold, so s*S_s(s*omega) differs
from S_1(omega).

(a) Decisive test: 9-site Heisenberg ring (Sz = +-1/2 ground doublets) plus a uniform field
h = 5e-9 * sum Sz, probe O = S^+_q. Symmetry(spatial=None): every Sz block is small and dense, so
energies are exact. At scale s = 1 the field splitting (5e-9) is inside the window; at s = 1000
it is 5e-6 and outside. Lorentzian S scales as S_s(s w; s eta) = S_1(w; eta) / s in exact physics.
(b) Informational: 13-site ring, translations only, time reversal off, dense_max_dim=1 so the
k and -k ground blocks go through the k = 1 Lanczos lane; ground_manifold is printed versus the
scale s (the claim predicts it can drop when the first-pass error ~ (1e-7 ||H||)^2/gap > 1e-8).
Run in a subprocess (it once needed an environment override)."""

import signal
import subprocess
import sys
import numpy as np
import qed

signal.alarm(280)


def ring(N, s, h=0.0):
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 * s)
        if h:
            H.add_one_body(qed.OP_SZ, i, h * s)
    return H


N = 9
q = 2 * np.pi * 4 / N
O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SPLUS, j, np.exp(1j * q * j) / np.sqrt(N))
w = np.linspace(0.0, 3.0, 121)
eta = 0.05
sym = qed.Symmetry(spatial=None)
res = {}
try:
    for s in (1.0, 1000.0):
        r = qed.dynamics(ring(N, s, h=5e-9), O, s * w, eta=s * eta, sym=sym)
        res[s] = (int(r.ground_manifold), s * np.asarray(r.S[0]), float(r.e0))
        print(
            f"(a) s={s:g}: ground_manifold={res[s][0]} e0/s={res[s][2] / s:.12f} "
            f"sum S dw={float(np.sum(res[s][1]) * (w[1] - w[0])):.6f}"
        )
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE (a) raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

dS = float(np.max(np.abs(res[1.0][1] - res[1000.0][1])) / max(np.max(np.abs(res[1.0][1])), 1e-300))

# (b) informational, in a subprocess, with the dense crossover at 1.
code_b = r'''
import numpy as np, qed
N = 13
t = [(i + 1) % N for i in range(N)]
O = qed.Operator(N)
for j in range(N):
    O.add_one_body(qed.OP_SZ, j, np.exp(1j * np.pi * j * 6 / 13) / np.sqrt(N))
sym = qed.Symmetry(spatial=[t], point_group=False, time_reversal="off")
w = np.linspace(0.0, 3.0, 11)
for s in (1.0, 1e2, 1e3, 1e4, 1e5):
    H = qed.Operator(N)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 * s)
    try:
        r = qed.dynamics(H, O, s * w, eta=0.05 * s, sym=sym, dense_max_dim=1)
        print(f"(b) s={s:g}: ground_manifold={r.ground_manifold}", flush=True)
    except Exception as ex:
        print(f"(b) s={s:g}: raised {type(ex).__name__}: {str(ex)[:120]}", flush=True)
'''
try:
    p = subprocess.run([sys.executable, "-c", code_b], capture_output=True, text=True, timeout=150)
    for line in p.stdout.splitlines():
        if line.startswith("(b)"):
            print(line)
    if p.returncode != 0:
        print(f"(b) subprocess rc={p.returncode}: {p.stderr[-300:]}")
except Exception as ex:
    print(f"(b) not run: {type(ex).__name__}: {ex}")

g1, g2 = res[1.0][0], res[1000.0][0]
if g1 != g2 and dS > 1e-3:
    print(
        f"REPRO: CONFIRMED ground_manifold {g1} (s=1) vs {g2} (s=1000) for the same physics; "
        f"max rel diff of s*S_s(s w) vs S_1(w) = {dS:.3e}"
    )
else:
    print(f"REPRO: NOT_REPRODUCED ground_manifold {g1} vs {g2}; max rel diff {dS:.3e}")
