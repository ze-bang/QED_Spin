# AUDIT-ID: L2-numerics-03
# DEVICE: cpu
# SECONDS: 120
"""Claim: the certified GS-vector lane (solve_block_eigenpairs -> solve_gs_vector) accepts a vector
only if ||H u - E0 u|| <= kLgGsResidTol = 1e-8 in ABSOLUTE energy units (lg_internal.h:121,
lg_ground_state.cpp:255). For H with a large energy scale the attainable residual exceeds 1e-8,
the block is reported uncertified and qed.eigs(H, 1, vectors=True) raises, while the values-only
qed.eigs(H, 1) (relative 1e-7*scale gate) succeeds.

Model: 16-site Heisenberg ring scaled by s, Symmetry(spatial=None): the Sz = 0 block (12870 states)
is above the k = 1 dense floor (1600), so the vector goes through the FullCGS2 Lanczos lane with the
absolute residual guard. Reference: E0(s) = s * E0(1) from the values-only call at s = 1."""
import signal
import numpy as np
import qed

signal.alarm(280)
N = 16


def ring(s):
    H = qed.Operator(N, 0.5)
    for i in range(N):
        j = (i + 1) % N
        H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5 * s)
        H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0 * s)
    return H


sym = qed.Symmetry(spatial=None)
try:
    e1 = float(qed.eigs(ring(1.0), 1, sym=sym).energies[0])
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE values-only eigs at s=1 raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

rows = {}
for s in (1.0, 1e3, 1e6, 1e9):
    H = ring(s)
    try:
        ev = float(qed.eigs(H, 1, sym=sym).energies[0])
        vals = f"values ok, E0/s-E0(1)={ev / s - e1:+.2e}"
        vals_ok = abs(ev / s - e1) < 1e-8 * max(1.0, abs(e1))
    except Exception as ex:
        vals, vals_ok = f"values raised {type(ex).__name__}: {str(ex)[:100]}", False
    try:
        rv = qed.eigs(H, 1, sym=sym, vectors=True)
        vec, vec_ok = f"vectors ok, E0/s-E0(1)={float(rv.energies[0]) / s - e1:+.2e}", True
    except Exception as ex:
        vec, vec_ok = f"vectors raised {type(ex).__name__}: {str(ex)[:140]}", False
    rows[s] = (vals_ok, vec_ok)
    print(f"s={s:g}: {vals}; {vec}")

if not (rows[1.0][0] and rows[1.0][1]):
    print("REPRO: INCONCLUSIVE the unscaled control failed: " + str(rows[1.0]))
else:
    bad = [s for s, (v, w) in rows.items() if v and not w]
    if bad:
        print(f"REPRO: CONFIRMED eigs(vectors=True) refuses at scale(s) {bad} while values-only eigs succeeds")
    else:
        print("REPRO: NOT_REPRODUCED vectors=True succeeded wherever values-only did: " + str(rows))
